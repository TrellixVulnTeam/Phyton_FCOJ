# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""This module contains the manager for handler methods used by the callback client"""
import logging
import threading
import abc
from azure.iot.device.common import handle_exceptions
from azure.iot.device.iothub.client_event import (
    CONNECTION_STATE_CHANGE,
    NEW_SASTOKEN_REQUIRED,
    BACKGROUND_EXCEPTION,
)
import concurrent.futures

logger = logging.getLogger(__name__)

# Receiver Handlers
MESSAGE = "_on_message_received"
METHOD = "_on_method_request_received"
TWIN_DP_PATCH = "_on_twin_desired_properties_patch_received"

# Client Event Handler Runner
CLIENT_EVENT = "client_event"
# Client Event Names
client_events = [CONNECTION_STATE_CHANGE, NEW_SASTOKEN_REQUIRED, BACKGROUND_EXCEPTION]


class HandlerManagerException(Exception):
    """An exception raised by a HandlerManager"""

    pass


class HandlerRunnerKillerSentinel(object):
    """An object that functions according to the sentinel design pattern.
    Insert into an Inbox in order to indicate that the Handler Runner associated with that
    Inbox should be stopped.
    """

    pass


class AbstractHandlerManager(abc.ABC):
    """Partial class that defines handler manager functionality shared between sync/async"""

    def __init__(self, inbox_manager):
        self._inbox_manager = inbox_manager

        self._receiver_handler_runners = {
            MESSAGE: None,
            METHOD: None,
            TWIN_DP_PATCH: None,
        }
        self._client_event_runner = None

        # Receiver handlers (Each will have it's own runner)
        self._on_message_received = None
        self._on_method_request_received = None
        self._on_twin_desired_properties_patch_received = None

        # Client Event handlers (Share a single Client Event runner)
        self._on_connection_state_change = None
        self._on_new_sastoken_required = None
        self._on_background_exception = None

        # As mentioned above, Receiver handlers each get their own runner. This is because it is
        # reasonably possible that many receives (and many different types of receives) can be
        # happening in a very short time. This is because we want to be able to support processing
        # multiple receives simultaneously, but also protect different receive invocations from
        # being slowed by a poorly written handler of a different type. Thus each Receiver handler
        # gets its own unique runner that cannot end up blocked by other handlers.
        #
        # Client Events on the other hand are generated by the client itself rather than
        # unsolicited data received over the wire. As these are relatively uncommon, we don't
        # expect multiple Client Events of the same type to really start queueing up, so there
        # isn't much justification for each type of event getting it's own dedicated runner.
        # Furthermore, there is much less concern over one inefficient or slow handler blocking
        # execution of others due to this infrequency.
        #
        # However, there are other differences between these handler classes as well. Receiver
        # handlers will always be invoked with a single argument - the received data structure.
        # Client Event handlers on the other hand are more flexible - they may be invoked with
        # different numbers of arguments depending on the handler, from none to multiple. This
        # is to keep design space open for more complex Client Events in the future.
        #
        # Finally, it is possible to stop ONLY the receiver handlers via the .stop() method
        # if desired. This is useful because while receiver handlers should be stopped when the
        # client disconnects, Client Events can still occur while the client is disconnected
        # as they are propagated from the client itself rather than received data.

    def _get_inbox_for_receive_handler(self, handler_name):
        """Retrieve the inbox relevant to the handler"""
        if handler_name == METHOD:
            return self._inbox_manager.get_method_request_inbox()
        elif handler_name == TWIN_DP_PATCH:
            return self._inbox_manager.get_twin_patch_inbox()
        elif handler_name == MESSAGE:
            return self._inbox_manager.get_unified_message_inbox()
        else:
            return None

    def _get_handler_for_client_event(self, event_name):
        """Retrieve the handler relevant to the event"""
        if event_name == NEW_SASTOKEN_REQUIRED:
            return self._on_new_sastoken_required
        elif event_name == CONNECTION_STATE_CHANGE:
            return self._on_connection_state_change
        elif event_name == BACKGROUND_EXCEPTION:
            return self._on_background_exception
        else:
            return None

    @abc.abstractmethod
    def _receiver_handler_runner(self, inbox, handler_name):
        """Run infinite loop that waits for an inbox to receive an object from it, then calls
        the handler with that object
        """
        pass

    @abc.abstractmethod
    def _client_event_handler_runner(self, handler_name):
        """Run infinite loop that waits for the client event inbox to receive an event from it,
        then calls the handler that corresponds to that event
        """
        pass

    @abc.abstractmethod
    def _start_handler_runner(self, handler_name):
        """Create, and store a handler runner"""
        pass

    @abc.abstractmethod
    def _stop_receiver_handler_runner(self, handler_name):
        """Cancel and remove a handler runner"""
        pass

    @abc.abstractmethod
    def _stop_client_event_handler_runner(self):
        """Cancel the client event handler runner"""
        pass

    def _generic_receiver_handler_setter(self, handler_name, new_handler):
        """Set a handler"""
        curr_handler = getattr(self, handler_name)
        if new_handler is not None and curr_handler is None:
            # Create runner, set handler
            logger.debug("Creating new handler runner for handler: {}".format(handler_name))
            setattr(self, handler_name, new_handler)
            self._start_handler_runner(handler_name)
        elif new_handler is None and curr_handler is not None:
            # Cancel runner, remove handler
            logger.debug("Removing handler runner for handler: {}".format(handler_name))
            self._stop_receiver_handler_runner(handler_name)
            setattr(self, handler_name, new_handler)
        else:
            # Update handler, no need to change runner
            logger.debug("Updating set handler: {}".format(handler_name))
            setattr(self, handler_name, new_handler)

    @staticmethod
    def _generate_callback_for_handler(handler_name):
        """Define a callback that can handle errors during handler execution"""

        def handler_callback(future):
            try:
                e = future.exception(timeout=0)
            except Exception as raised_e:
                # This shouldn't happen because cancellation or timeout shouldn't occur...
                # But just in case...
                new_err = HandlerManagerException(
                    "HANDLER ({}): Unable to retrieve exception data from incomplete invocation".format(
                        handler_name
                    )
                )
                new_err.__cause__ = raised_e
                handle_exceptions.handle_background_exception(new_err)
            else:
                if e:
                    new_err = HandlerManagerException(
                        "HANDLER ({}): Error during invocation".format(handler_name),
                    )
                    new_err.__cause__ = e
                    handle_exceptions.handle_background_exception(new_err)
                else:
                    logger.debug(
                        "HANDLER ({}): Successfully completed invocation".format(handler_name)
                    )

        return handler_callback

    def stop(self, receiver_handlers_only=False):
        """Stop the process of invoking handlers in response to events.
        All pending items will be handled prior to stoppage.
        """
        # Stop receiver handlers
        for handler_name in self._receiver_handler_runners:
            if self._receiver_handler_runners[handler_name] is not None:
                self._stop_receiver_handler_runner(handler_name)

        # Stop the client event handler (if instructed)
        if not receiver_handlers_only and self._client_event_runner is not None:
            self._stop_client_event_handler_runner()

    def ensure_running(self):
        """Ensure the process of invoking handlers in response to events is running"""
        # Ensure any receiver handler set on the manager has a corresponding handler runner running
        for handler_name in self._receiver_handler_runners:
            if (
                self._receiver_handler_runners[handler_name] is None
                and getattr(self, handler_name) is not None
            ):
                self._start_handler_runner(handler_name)

        # Ensure client event handler runner is running if at least one client event handler is set
        # on the manager
        if self._client_event_runner is None:
            for event in client_events:
                handler = self._get_handler_for_client_event(event)
                if handler is not None:
                    self._start_handler_runner(CLIENT_EVENT)
                    break

    # ~~~Receiver Handlers~~~
    # Setting a receiver handler will start a dedicated runner for that handler
    # Removing a receiver handler will stop the dedicated runner for that handler
    @property
    def on_message_received(self):
        return self._on_message_received

    @on_message_received.setter
    def on_message_received(self, value):
        self._generic_receiver_handler_setter(MESSAGE, value)

    @property
    def on_method_request_received(self):
        return self._on_method_request_received

    @on_method_request_received.setter
    def on_method_request_received(self, value):
        self._generic_receiver_handler_setter(METHOD, value)

    @property
    def on_twin_desired_properties_patch_received(self):
        return self._on_twin_desired_properties_patch_received

    @on_twin_desired_properties_patch_received.setter
    def on_twin_desired_properties_patch_received(self, value):
        self._generic_receiver_handler_setter(TWIN_DP_PATCH, value)

    # ~~~Client Event Handlers~~~
    # Setting any client event handler will start the shared client event handler runner
    # Removing handlers will NOT stop the client event handler runner - you must use .stop()
    # Stopping when all client event handlers are removed could be added if necessary.
    @property
    def on_connection_state_change(self):
        return self._on_connection_state_change

    @on_connection_state_change.setter
    def on_connection_state_change(self, value):
        self._on_connection_state_change = value
        if self._client_event_runner is None:
            self._start_handler_runner(CLIENT_EVENT)

    @property
    def on_new_sastoken_required(self):
        return self._on_new_sastoken_required

    @on_new_sastoken_required.setter
    def on_new_sastoken_required(self, value):
        self._on_new_sastoken_required = value
        if self._client_event_runner is None:
            self._start_handler_runner(CLIENT_EVENT)

    @property
    def on_background_exception(self):
        return self._on_background_exception

    @on_background_exception.setter
    def on_background_exception(self, value):
        self._on_background_exception = value
        if self._client_event_runner is None:
            self._start_handler_runner(CLIENT_EVENT)

    # ~~~Other Properties~~~
    @property
    def handling_client_events(self):
        """Indicates if the HandlerManager is currently capable of resolving ClientEvents"""
        # This client event runner is only running if at least one handler for client events has
        # been set. If none have been set, it is dangerous to add items to the client event inbox
        # as none will ever be retrieved due to no runner process occurring, thus the need for this
        # check.
        #
        # The ideal solution would be to always keep the client event runner running, but this
        # could break older customer code due to older APIs on the customer-facing clients. It is
        # unfortunate that something related to an API has seeped into this internal and ideally
        # isolated module, but the needs of the client design have influenced the design of this
        # manager (by only starting the runner when a handler is set), so the mitigation must also
        # be located in this module.
        if self._client_event_runner is None:
            return False
        else:
            return True


class SyncHandlerManager(AbstractHandlerManager):
    """Handler manager for use with synchronous clients"""

    def _receiver_handler_runner(self, inbox, handler_name):
        """Run infinite loop that waits for an inbox to receive an object from it, then calls
        the handler with that object
        """
        logger.debug("HANDLER RUNNER ({}): Starting runner".format(handler_name))
        _handler_callback = self._generate_callback_for_handler(handler_name)

        # Run the handler in a threadpool, so that it cannot block other handlers (from a different task),
        # or the main client thread. The number of worker threads forms an upper bound on how many instances
        # of the same handler can be running simultaneously.
        tpe = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        while True:
            handler_arg = inbox.get()
            if isinstance(handler_arg, HandlerRunnerKillerSentinel):
                # Exit the runner when a HandlerRunnerKillerSentinel is found
                logger.debug(
                    "HANDLER RUNNER ({}): HandlerRunnerKillerSentinel found in inbox. Exiting.".format(
                        handler_name
                    )
                )
                tpe.shutdown()
                break
            # NOTE: we MUST use getattr here using the handler name, as opposed to directly passing
            # the handler in order for the handler to be able to be updated without cancelling
            # the running task created for this coroutine
            handler = getattr(self, handler_name)
            logger.debug("HANDLER RUNNER ({}): Invoking handler".format(handler_name))
            fut = tpe.submit(handler, handler_arg)
            fut.add_done_callback(_handler_callback)
            # Free up this object so the garbage collector can free it if necessary. If we don't
            # do this, we end up keeping this object alive until the next event arrives, which
            # might be a long time. Tests would flag this as a memory leak if that happened.
            del handler_arg

    def _client_event_handler_runner(self):
        """Run infinite loop that waits for the client event inbox to receive an event from it,
        then calls the handler that corresponds to that event
        """
        logger.debug("HANDLER RUNNER (CLIENT EVENT): Starting runner")
        _handler_callback = self._generate_callback_for_handler("CLIENT_EVENT")

        tpe = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        event_inbox = self._inbox_manager.get_client_event_inbox()
        while True:
            event = event_inbox.get()
            if isinstance(event, HandlerRunnerKillerSentinel):
                # Exit the runner when a HandlerRunnerKillerSentinel is found
                logger.debug(
                    "HANDLER RUNNER (CLIENT EVENT): HandlerRunnerKillerSentinel found in event queue. Exiting."
                )
                tpe.shutdown()
                break
            handler = self._get_handler_for_client_event(event.name)
            if handler is not None:
                logger.debug(
                    "HANDLER RUNNER (CLIENT EVENT): {} event received. Invoking {} handler".format(
                        event, handler
                    )
                )
                fut = tpe.submit(handler, *event.args_for_user)
                fut.add_done_callback(_handler_callback)
                # Free up this object so the garbage collector can free it if necessary. If we don't
                # do this, we end up keeping this object alive until the next event arrives, which
                # might be a long time. Tests would flag this as a memory leak if that happened.
                del event
            else:
                logger.debug(
                    "No handler for event {} set. Skipping handler invocation".format(event)
                )

    def _start_handler_runner(self, handler_name):
        """Start and store a handler runner thread"""
        # Client Event handler flow
        if handler_name == CLIENT_EVENT:
            if self._client_event_runner is not None:
                # This branch of code should NOT be reachable due to checks prior to the invocation
                # of this method. The branch exists for safety.
                raise HandlerManagerException(
                    "Cannot create thread for handler runner: {}. Runner thread already exists".format(
                        handler_name
                    )
                )
            # Client events share a handler
            thread = threading.Thread(target=self._client_event_handler_runner)
            # Store the thread
            self._client_event_runner = thread

        # Receiver handler flow
        else:
            if self._receiver_handler_runners[handler_name] is not None:
                # This branch of code should NOT be reachable due to checks prior to the invocation
                # of this method. The branch exists for safety.
                raise HandlerManagerException(
                    "Cannot create thread for handler runner: {}. Runner thread already exists".format(
                        handler_name
                    )
                )
            inbox = self._get_inbox_for_receive_handler(handler_name)
            # Each receiver handler gets its own runner
            thread = threading.Thread(
                target=self._receiver_handler_runner, args=[inbox, handler_name]
            )
            # Store the thread
            self._receiver_handler_runners[handler_name] = thread

        # NOTE: It would be nice to have some kind of mechanism for making sure this thread
        # doesn't crash or raise errors, but it would require significant extra infrastructure
        # and an exception in here isn't supposed to happen anyway. Perhaps it could be added
        # later if truly necessary
        thread.daemon = True  # Don't block program exit
        thread.start()

    def _stop_receiver_handler_runner(self, handler_name):
        """Stop and remove a handler runner thread.
        All pending items in the corresponding inbox will be handled by the handler before stoppage.
        """
        logger.debug(
            "Adding HandlerRunnerKillerSentinel to inbox corresponding to {} handler runner".format(
                handler_name
            )
        )
        inbox = self._get_inbox_for_receive_handler(handler_name)
        inbox.put(HandlerRunnerKillerSentinel())

        # Wait for Handler Runner to end due to the sentinel
        logger.debug("Waiting for {} handler runner to exit...".format(handler_name))
        thread = self._receiver_handler_runners[handler_name]
        thread.join()
        self._receiver_handler_runners[handler_name] = None
        logger.debug("Handler runner for {} has been stopped".format(handler_name))

    def _stop_client_event_handler_runner(self):
        """Stop and remove a handler runner thread.
        All pending items in the client event queue will be handled by handlers (if they exist)
        before stoppage.
        """
        logger.debug("Adding HandlerRunnerKillerSentinel to client event queue")
        event_inbox = self._inbox_manager.get_client_event_inbox()
        event_inbox.put(HandlerRunnerKillerSentinel())

        # Wait for Handler Runner to end due to the stop command
        logger.debug("Waiting for client event handler runner to exit...")
        thread = self._client_event_runner
        thread.join()
        self._client_event_runner = None
        logger.debug("Handler runner for client events has been stopped")