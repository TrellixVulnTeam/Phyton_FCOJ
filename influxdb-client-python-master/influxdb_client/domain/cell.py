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


class Cell(object):
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
        'links': 'CellLinks',
        'x': 'int',
        'y': 'int',
        'w': 'int',
        'h': 'int',
        'view_id': 'str'
    }

    attribute_map = {
        'id': 'id',
        'links': 'links',
        'x': 'x',
        'y': 'y',
        'w': 'w',
        'h': 'h',
        'view_id': 'viewID'
    }

    def __init__(self, id=None, links=None, x=None, y=None, w=None, h=None, view_id=None):  # noqa: E501,D401,D403
        """Cell - a model defined in OpenAPI."""  # noqa: E501
        self._id = None
        self._links = None
        self._x = None
        self._y = None
        self._w = None
        self._h = None
        self._view_id = None
        self.discriminator = None

        if id is not None:
            self.id = id
        if links is not None:
            self.links = links
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if w is not None:
            self.w = w
        if h is not None:
            self.h = h
        if view_id is not None:
            self.view_id = view_id

    @property
    def id(self):
        """Get the id of this Cell.

        :return: The id of this Cell.
        :rtype: str
        """  # noqa: E501
        return self._id

    @id.setter
    def id(self, id):
        """Set the id of this Cell.

        :param id: The id of this Cell.
        :type: str
        """  # noqa: E501
        self._id = id

    @property
    def links(self):
        """Get the links of this Cell.

        :return: The links of this Cell.
        :rtype: CellLinks
        """  # noqa: E501
        return self._links

    @links.setter
    def links(self, links):
        """Set the links of this Cell.

        :param links: The links of this Cell.
        :type: CellLinks
        """  # noqa: E501
        self._links = links

    @property
    def x(self):
        """Get the x of this Cell.

        :return: The x of this Cell.
        :rtype: int
        """  # noqa: E501
        return self._x

    @x.setter
    def x(self, x):
        """Set the x of this Cell.

        :param x: The x of this Cell.
        :type: int
        """  # noqa: E501
        self._x = x

    @property
    def y(self):
        """Get the y of this Cell.

        :return: The y of this Cell.
        :rtype: int
        """  # noqa: E501
        return self._y

    @y.setter
    def y(self, y):
        """Set the y of this Cell.

        :param y: The y of this Cell.
        :type: int
        """  # noqa: E501
        self._y = y

    @property
    def w(self):
        """Get the w of this Cell.

        :return: The w of this Cell.
        :rtype: int
        """  # noqa: E501
        return self._w

    @w.setter
    def w(self, w):
        """Set the w of this Cell.

        :param w: The w of this Cell.
        :type: int
        """  # noqa: E501
        self._w = w

    @property
    def h(self):
        """Get the h of this Cell.

        :return: The h of this Cell.
        :rtype: int
        """  # noqa: E501
        return self._h

    @h.setter
    def h(self, h):
        """Set the h of this Cell.

        :param h: The h of this Cell.
        :type: int
        """  # noqa: E501
        self._h = h

    @property
    def view_id(self):
        """Get the view_id of this Cell.

        The reference to a view from the views API.

        :return: The view_id of this Cell.
        :rtype: str
        """  # noqa: E501
        return self._view_id

    @view_id.setter
    def view_id(self, view_id):
        """Set the view_id of this Cell.

        The reference to a view from the views API.

        :param view_id: The view_id of this Cell.
        :type: str
        """  # noqa: E501
        self._view_id = view_id

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
        if not isinstance(other, Cell):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Return true if both objects are not equal."""
        return not self == other