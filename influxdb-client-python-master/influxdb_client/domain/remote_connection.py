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


class RemoteConnection(object):
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
        'name': 'str',
        'org_id': 'str',
        'description': 'str',
        'remote_url': 'str',
        'remote_org_id': 'str',
        'allow_insecure_tls': 'bool'
    }

    attribute_map = {
        'id': 'id',
        'name': 'name',
        'org_id': 'orgID',
        'description': 'description',
        'remote_url': 'remoteURL',
        'remote_org_id': 'remoteOrgID',
        'allow_insecure_tls': 'allowInsecureTLS'
    }

    def __init__(self, id=None, name=None, org_id=None, description=None, remote_url=None, remote_org_id=None, allow_insecure_tls=False):  # noqa: E501,D401,D403
        """RemoteConnection - a model defined in OpenAPI."""  # noqa: E501
        self._id = None
        self._name = None
        self._org_id = None
        self._description = None
        self._remote_url = None
        self._remote_org_id = None
        self._allow_insecure_tls = None
        self.discriminator = None

        self.id = id
        self.name = name
        self.org_id = org_id
        if description is not None:
            self.description = description
        self.remote_url = remote_url
        self.remote_org_id = remote_org_id
        self.allow_insecure_tls = allow_insecure_tls

    @property
    def id(self):
        """Get the id of this RemoteConnection.

        :return: The id of this RemoteConnection.
        :rtype: str
        """  # noqa: E501
        return self._id

    @id.setter
    def id(self, id):
        """Set the id of this RemoteConnection.

        :param id: The id of this RemoteConnection.
        :type: str
        """  # noqa: E501
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501
        self._id = id

    @property
    def name(self):
        """Get the name of this RemoteConnection.

        :return: The name of this RemoteConnection.
        :rtype: str
        """  # noqa: E501
        return self._name

    @name.setter
    def name(self, name):
        """Set the name of this RemoteConnection.

        :param name: The name of this RemoteConnection.
        :type: str
        """  # noqa: E501
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501
        self._name = name

    @property
    def org_id(self):
        """Get the org_id of this RemoteConnection.

        :return: The org_id of this RemoteConnection.
        :rtype: str
        """  # noqa: E501
        return self._org_id

    @org_id.setter
    def org_id(self, org_id):
        """Set the org_id of this RemoteConnection.

        :param org_id: The org_id of this RemoteConnection.
        :type: str
        """  # noqa: E501
        if org_id is None:
            raise ValueError("Invalid value for `org_id`, must not be `None`")  # noqa: E501
        self._org_id = org_id

    @property
    def description(self):
        """Get the description of this RemoteConnection.

        :return: The description of this RemoteConnection.
        :rtype: str
        """  # noqa: E501
        return self._description

    @description.setter
    def description(self, description):
        """Set the description of this RemoteConnection.

        :param description: The description of this RemoteConnection.
        :type: str
        """  # noqa: E501
        self._description = description

    @property
    def remote_url(self):
        """Get the remote_url of this RemoteConnection.

        :return: The remote_url of this RemoteConnection.
        :rtype: str
        """  # noqa: E501
        return self._remote_url

    @remote_url.setter
    def remote_url(self, remote_url):
        """Set the remote_url of this RemoteConnection.

        :param remote_url: The remote_url of this RemoteConnection.
        :type: str
        """  # noqa: E501
        if remote_url is None:
            raise ValueError("Invalid value for `remote_url`, must not be `None`")  # noqa: E501
        self._remote_url = remote_url

    @property
    def remote_org_id(self):
        """Get the remote_org_id of this RemoteConnection.

        :return: The remote_org_id of this RemoteConnection.
        :rtype: str
        """  # noqa: E501
        return self._remote_org_id

    @remote_org_id.setter
    def remote_org_id(self, remote_org_id):
        """Set the remote_org_id of this RemoteConnection.

        :param remote_org_id: The remote_org_id of this RemoteConnection.
        :type: str
        """  # noqa: E501
        if remote_org_id is None:
            raise ValueError("Invalid value for `remote_org_id`, must not be `None`")  # noqa: E501
        self._remote_org_id = remote_org_id

    @property
    def allow_insecure_tls(self):
        """Get the allow_insecure_tls of this RemoteConnection.

        :return: The allow_insecure_tls of this RemoteConnection.
        :rtype: bool
        """  # noqa: E501
        return self._allow_insecure_tls

    @allow_insecure_tls.setter
    def allow_insecure_tls(self, allow_insecure_tls):
        """Set the allow_insecure_tls of this RemoteConnection.

        :param allow_insecure_tls: The allow_insecure_tls of this RemoteConnection.
        :type: bool
        """  # noqa: E501
        if allow_insecure_tls is None:
            raise ValueError("Invalid value for `allow_insecure_tls`, must not be `None`")  # noqa: E501
        self._allow_insecure_tls = allow_insecure_tls

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
        if not isinstance(other, RemoteConnection):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Return true if both objects are not equal."""
        return not self == other
