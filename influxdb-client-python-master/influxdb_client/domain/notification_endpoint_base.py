# coding: utf-8

"""
InfluxDB OSS API Service.

The InfluxDB v2 API provides a programmatic interface for all interactions with InfluxDB. Access the InfluxDB API using the `/api/v2/` endpoint.   # noqa: E501

OpenAPI spec version: 2.0.0
Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six


class NotificationEndpointBase(object):
    """NOTE: This class is auto generated by OpenAPI Generator.

    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'id': 'str',
        'org_id': 'str',
        'user_id': 'str',
        'created_at': 'datetime',
        'updated_at': 'datetime',
        'description': 'str',
        'name': 'str',
        'status': 'str',
        'labels': 'list[Label]',
        'links': 'NotificationEndpointBaseLinks',
        'type': 'NotificationEndpointType'
    }

    attribute_map = {
        'id': 'id',
        'org_id': 'orgID',
        'user_id': 'userID',
        'created_at': 'createdAt',
        'updated_at': 'updatedAt',
        'description': 'description',
        'name': 'name',
        'status': 'status',
        'labels': 'labels',
        'links': 'links',
        'type': 'type'
    }

    def __init__(self, id=None, org_id=None, user_id=None, created_at=None, updated_at=None, description=None, name=None, status='active', labels=None, links=None, type=None):  # noqa: E501,D401,D403
        """NotificationEndpointBase - a model defined in OpenAPI."""  # noqa: E501
        self._id = None
        self._org_id = None
        self._user_id = None
        self._created_at = None
        self._updated_at = None
        self._description = None
        self._name = None
        self._status = None
        self._labels = None
        self._links = None
        self._type = None
        self.discriminator = None

        if id is not None:
            self.id = id
        if org_id is not None:
            self.org_id = org_id
        if user_id is not None:
            self.user_id = user_id
        if created_at is not None:
            self.created_at = created_at
        if updated_at is not None:
            self.updated_at = updated_at
        if description is not None:
            self.description = description
        self.name = name
        if status is not None:
            self.status = status
        if labels is not None:
            self.labels = labels
        if links is not None:
            self.links = links
        self.type = type

    @property
    def id(self):
        """Get the id of this NotificationEndpointBase.

        :return: The id of this NotificationEndpointBase.
        :rtype: str
        """  # noqa: E501
        return self._id

    @id.setter
    def id(self, id):
        """Set the id of this NotificationEndpointBase.

        :param id: The id of this NotificationEndpointBase.
        :type: str
        """  # noqa: E501
        self._id = id

    @property
    def org_id(self):
        """Get the org_id of this NotificationEndpointBase.

        :return: The org_id of this NotificationEndpointBase.
        :rtype: str
        """  # noqa: E501
        return self._org_id

    @org_id.setter
    def org_id(self, org_id):
        """Set the org_id of this NotificationEndpointBase.

        :param org_id: The org_id of this NotificationEndpointBase.
        :type: str
        """  # noqa: E501
        self._org_id = org_id

    @property
    def user_id(self):
        """Get the user_id of this NotificationEndpointBase.

        :return: The user_id of this NotificationEndpointBase.
        :rtype: str
        """  # noqa: E501
        return self._user_id

    @user_id.setter
    def user_id(self, user_id):
        """Set the user_id of this NotificationEndpointBase.

        :param user_id: The user_id of this NotificationEndpointBase.
        :type: str
        """  # noqa: E501
        self._user_id = user_id

    @property
    def created_at(self):
        """Get the created_at of this NotificationEndpointBase.

        :return: The created_at of this NotificationEndpointBase.
        :rtype: datetime
        """  # noqa: E501
        return self._created_at

    @created_at.setter
    def created_at(self, created_at):
        """Set the created_at of this NotificationEndpointBase.

        :param created_at: The created_at of this NotificationEndpointBase.
        :type: datetime
        """  # noqa: E501
        self._created_at = created_at

    @property
    def updated_at(self):
        """Get the updated_at of this NotificationEndpointBase.

        :return: The updated_at of this NotificationEndpointBase.
        :rtype: datetime
        """  # noqa: E501
        return self._updated_at

    @updated_at.setter
    def updated_at(self, updated_at):
        """Set the updated_at of this NotificationEndpointBase.

        :param updated_at: The updated_at of this NotificationEndpointBase.
        :type: datetime
        """  # noqa: E501
        self._updated_at = updated_at

    @property
    def description(self):
        """Get the description of this NotificationEndpointBase.

        An optional description of the notification endpoint.

        :return: The description of this NotificationEndpointBase.
        :rtype: str
        """  # noqa: E501
        return self._description

    @description.setter
    def description(self, description):
        """Set the description of this NotificationEndpointBase.

        An optional description of the notification endpoint.

        :param description: The description of this NotificationEndpointBase.
        :type: str
        """  # noqa: E501
        self._description = description

    @property
    def name(self):
        """Get the name of this NotificationEndpointBase.

        :return: The name of this NotificationEndpointBase.
        :rtype: str
        """  # noqa: E501
        return self._name

    @name.setter
    def name(self, name):
        """Set the name of this NotificationEndpointBase.

        :param name: The name of this NotificationEndpointBase.
        :type: str
        """  # noqa: E501
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501
        self._name = name

    @property
    def status(self):
        """Get the status of this NotificationEndpointBase.

        The status of the endpoint.

        :return: The status of this NotificationEndpointBase.
        :rtype: str
        """  # noqa: E501
        return self._status

    @status.setter
    def status(self, status):
        """Set the status of this NotificationEndpointBase.

        The status of the endpoint.

        :param status: The status of this NotificationEndpointBase.
        :type: str
        """  # noqa: E501
        self._status = status

    @property
    def labels(self):
        """Get the labels of this NotificationEndpointBase.

        :return: The labels of this NotificationEndpointBase.
        :rtype: list[Label]
        """  # noqa: E501
        return self._labels

    @labels.setter
    def labels(self, labels):
        """Set the labels of this NotificationEndpointBase.

        :param labels: The labels of this NotificationEndpointBase.
        :type: list[Label]
        """  # noqa: E501
        self._labels = labels

    @property
    def links(self):
        """Get the links of this NotificationEndpointBase.

        :return: The links of this NotificationEndpointBase.
        :rtype: NotificationEndpointBaseLinks
        """  # noqa: E501
        return self._links

    @links.setter
    def links(self, links):
        """Set the links of this NotificationEndpointBase.

        :param links: The links of this NotificationEndpointBase.
        :type: NotificationEndpointBaseLinks
        """  # noqa: E501
        self._links = links

    @property
    def type(self):
        """Get the type of this NotificationEndpointBase.

        :return: The type of this NotificationEndpointBase.
        :rtype: NotificationEndpointType
        """  # noqa: E501
        return self._type

    @type.setter
    def type(self, type):
        """Set the type of this NotificationEndpointBase.

        :param type: The type of this NotificationEndpointBase.
        :type: NotificationEndpointType
        """  # noqa: E501
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")  # noqa: E501
        self._type = type

    def to_dict(self):
        """Return the model properties as a dict."""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Return the string representation of the model."""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`."""
        return self.to_str()

    def __eq__(self, other):
        """Return true if both objects are equal."""
        if not isinstance(other, NotificationEndpointBase):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Return true if both objects are not equal."""
        return not self == other