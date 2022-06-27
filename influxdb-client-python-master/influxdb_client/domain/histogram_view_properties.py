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
from influxdb_client.domain.view_properties import ViewProperties


class HistogramViewProperties(ViewProperties):
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
        'type': 'str',
        'queries': 'list[DashboardQuery]',
        'colors': 'list[DashboardColor]',
        'shape': 'str',
        'note': 'str',
        'show_note_when_empty': 'bool',
        'x_column': 'str',
        'fill_columns': 'list[str]',
        'x_domain': 'list[float]',
        'x_axis_label': 'str',
        'position': 'str',
        'bin_count': 'int',
        'legend_colorize_rows': 'bool',
        'legend_hide': 'bool',
        'legend_opacity': 'float',
        'legend_orientation_threshold': 'int'
    }

    attribute_map = {
        'type': 'type',
        'queries': 'queries',
        'colors': 'colors',
        'shape': 'shape',
        'note': 'note',
        'show_note_when_empty': 'showNoteWhenEmpty',
        'x_column': 'xColumn',
        'fill_columns': 'fillColumns',
        'x_domain': 'xDomain',
        'x_axis_label': 'xAxisLabel',
        'position': 'position',
        'bin_count': 'binCount',
        'legend_colorize_rows': 'legendColorizeRows',
        'legend_hide': 'legendHide',
        'legend_opacity': 'legendOpacity',
        'legend_orientation_threshold': 'legendOrientationThreshold'
    }

    def __init__(self, type=None, queries=None, colors=None, shape=None, note=None, show_note_when_empty=None, x_column=None, fill_columns=None, x_domain=None, x_axis_label=None, position=None, bin_count=None, legend_colorize_rows=None, legend_hide=None, legend_opacity=None, legend_orientation_threshold=None):  # noqa: E501,D401,D403
        """HistogramViewProperties - a model defined in OpenAPI."""  # noqa: E501
        ViewProperties.__init__(self)  # noqa: E501

        self._type = None
        self._queries = None
        self._colors = None
        self._shape = None
        self._note = None
        self._show_note_when_empty = None
        self._x_column = None
        self._fill_columns = None
        self._x_domain = None
        self._x_axis_label = None
        self._position = None
        self._bin_count = None
        self._legend_colorize_rows = None
        self._legend_hide = None
        self._legend_opacity = None
        self._legend_orientation_threshold = None
        self.discriminator = None

        self.type = type
        self.queries = queries
        self.colors = colors
        self.shape = shape
        self.note = note
        self.show_note_when_empty = show_note_when_empty
        self.x_column = x_column
        self.fill_columns = fill_columns
        self.x_domain = x_domain
        self.x_axis_label = x_axis_label
        self.position = position
        self.bin_count = bin_count
        if legend_colorize_rows is not None:
            self.legend_colorize_rows = legend_colorize_rows
        if legend_hide is not None:
            self.legend_hide = legend_hide
        if legend_opacity is not None:
            self.legend_opacity = legend_opacity
        if legend_orientation_threshold is not None:
            self.legend_orientation_threshold = legend_orientation_threshold

    @property
    def type(self):
        """Get the type of this HistogramViewProperties.

        :return: The type of this HistogramViewProperties.
        :rtype: str
        """  # noqa: E501
        return self._type

    @type.setter
    def type(self, type):
        """Set the type of this HistogramViewProperties.

        :param type: The type of this HistogramViewProperties.
        :type: str
        """  # noqa: E501
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")  # noqa: E501
        self._type = type

    @property
    def queries(self):
        """Get the queries of this HistogramViewProperties.

        :return: The queries of this HistogramViewProperties.
        :rtype: list[DashboardQuery]
        """  # noqa: E501
        return self._queries

    @queries.setter
    def queries(self, queries):
        """Set the queries of this HistogramViewProperties.

        :param queries: The queries of this HistogramViewProperties.
        :type: list[DashboardQuery]
        """  # noqa: E501
        if queries is None:
            raise ValueError("Invalid value for `queries`, must not be `None`")  # noqa: E501
        self._queries = queries

    @property
    def colors(self):
        """Get the colors of this HistogramViewProperties.

        Colors define color encoding of data into a visualization

        :return: The colors of this HistogramViewProperties.
        :rtype: list[DashboardColor]
        """  # noqa: E501
        return self._colors

    @colors.setter
    def colors(self, colors):
        """Set the colors of this HistogramViewProperties.

        Colors define color encoding of data into a visualization

        :param colors: The colors of this HistogramViewProperties.
        :type: list[DashboardColor]
        """  # noqa: E501
        if colors is None:
            raise ValueError("Invalid value for `colors`, must not be `None`")  # noqa: E501
        self._colors = colors

    @property
    def shape(self):
        """Get the shape of this HistogramViewProperties.

        :return: The shape of this HistogramViewProperties.
        :rtype: str
        """  # noqa: E501
        return self._shape

    @shape.setter
    def shape(self, shape):
        """Set the shape of this HistogramViewProperties.

        :param shape: The shape of this HistogramViewProperties.
        :type: str
        """  # noqa: E501
        if shape is None:
            raise ValueError("Invalid value for `shape`, must not be `None`")  # noqa: E501
        self._shape = shape

    @property
    def note(self):
        """Get the note of this HistogramViewProperties.

        :return: The note of this HistogramViewProperties.
        :rtype: str
        """  # noqa: E501
        return self._note

    @note.setter
    def note(self, note):
        """Set the note of this HistogramViewProperties.

        :param note: The note of this HistogramViewProperties.
        :type: str
        """  # noqa: E501
        if note is None:
            raise ValueError("Invalid value for `note`, must not be `None`")  # noqa: E501
        self._note = note

    @property
    def show_note_when_empty(self):
        """Get the show_note_when_empty of this HistogramViewProperties.

        If true, will display note when empty

        :return: The show_note_when_empty of this HistogramViewProperties.
        :rtype: bool
        """  # noqa: E501
        return self._show_note_when_empty

    @show_note_when_empty.setter
    def show_note_when_empty(self, show_note_when_empty):
        """Set the show_note_when_empty of this HistogramViewProperties.

        If true, will display note when empty

        :param show_note_when_empty: The show_note_when_empty of this HistogramViewProperties.
        :type: bool
        """  # noqa: E501
        if show_note_when_empty is None:
            raise ValueError("Invalid value for `show_note_when_empty`, must not be `None`")  # noqa: E501
        self._show_note_when_empty = show_note_when_empty

    @property
    def x_column(self):
        """Get the x_column of this HistogramViewProperties.

        :return: The x_column of this HistogramViewProperties.
        :rtype: str
        """  # noqa: E501
        return self._x_column

    @x_column.setter
    def x_column(self, x_column):
        """Set the x_column of this HistogramViewProperties.

        :param x_column: The x_column of this HistogramViewProperties.
        :type: str
        """  # noqa: E501
        if x_column is None:
            raise ValueError("Invalid value for `x_column`, must not be `None`")  # noqa: E501
        self._x_column = x_column

    @property
    def fill_columns(self):
        """Get the fill_columns of this HistogramViewProperties.

        :return: The fill_columns of this HistogramViewProperties.
        :rtype: list[str]
        """  # noqa: E501
        return self._fill_columns

    @fill_columns.setter
    def fill_columns(self, fill_columns):
        """Set the fill_columns of this HistogramViewProperties.

        :param fill_columns: The fill_columns of this HistogramViewProperties.
        :type: list[str]
        """  # noqa: E501
        if fill_columns is None:
            raise ValueError("Invalid value for `fill_columns`, must not be `None`")  # noqa: E501
        self._fill_columns = fill_columns

    @property
    def x_domain(self):
        """Get the x_domain of this HistogramViewProperties.

        :return: The x_domain of this HistogramViewProperties.
        :rtype: list[float]
        """  # noqa: E501
        return self._x_domain

    @x_domain.setter
    def x_domain(self, x_domain):
        """Set the x_domain of this HistogramViewProperties.

        :param x_domain: The x_domain of this HistogramViewProperties.
        :type: list[float]
        """  # noqa: E501
        if x_domain is None:
            raise ValueError("Invalid value for `x_domain`, must not be `None`")  # noqa: E501
        self._x_domain = x_domain

    @property
    def x_axis_label(self):
        """Get the x_axis_label of this HistogramViewProperties.

        :return: The x_axis_label of this HistogramViewProperties.
        :rtype: str
        """  # noqa: E501
        return self._x_axis_label

    @x_axis_label.setter
    def x_axis_label(self, x_axis_label):
        """Set the x_axis_label of this HistogramViewProperties.

        :param x_axis_label: The x_axis_label of this HistogramViewProperties.
        :type: str
        """  # noqa: E501
        if x_axis_label is None:
            raise ValueError("Invalid value for `x_axis_label`, must not be `None`")  # noqa: E501
        self._x_axis_label = x_axis_label

    @property
    def position(self):
        """Get the position of this HistogramViewProperties.

        :return: The position of this HistogramViewProperties.
        :rtype: str
        """  # noqa: E501
        return self._position

    @position.setter
    def position(self, position):
        """Set the position of this HistogramViewProperties.

        :param position: The position of this HistogramViewProperties.
        :type: str
        """  # noqa: E501
        if position is None:
            raise ValueError("Invalid value for `position`, must not be `None`")  # noqa: E501
        self._position = position

    @property
    def bin_count(self):
        """Get the bin_count of this HistogramViewProperties.

        :return: The bin_count of this HistogramViewProperties.
        :rtype: int
        """  # noqa: E501
        return self._bin_count

    @bin_count.setter
    def bin_count(self, bin_count):
        """Set the bin_count of this HistogramViewProperties.

        :param bin_count: The bin_count of this HistogramViewProperties.
        :type: int
        """  # noqa: E501
        if bin_count is None:
            raise ValueError("Invalid value for `bin_count`, must not be `None`")  # noqa: E501
        self._bin_count = bin_count

    @property
    def legend_colorize_rows(self):
        """Get the legend_colorize_rows of this HistogramViewProperties.

        :return: The legend_colorize_rows of this HistogramViewProperties.
        :rtype: bool
        """  # noqa: E501
        return self._legend_colorize_rows

    @legend_colorize_rows.setter
    def legend_colorize_rows(self, legend_colorize_rows):
        """Set the legend_colorize_rows of this HistogramViewProperties.

        :param legend_colorize_rows: The legend_colorize_rows of this HistogramViewProperties.
        :type: bool
        """  # noqa: E501
        self._legend_colorize_rows = legend_colorize_rows

    @property
    def legend_hide(self):
        """Get the legend_hide of this HistogramViewProperties.

        :return: The legend_hide of this HistogramViewProperties.
        :rtype: bool
        """  # noqa: E501
        return self._legend_hide

    @legend_hide.setter
    def legend_hide(self, legend_hide):
        """Set the legend_hide of this HistogramViewProperties.

        :param legend_hide: The legend_hide of this HistogramViewProperties.
        :type: bool
        """  # noqa: E501
        self._legend_hide = legend_hide

    @property
    def legend_opacity(self):
        """Get the legend_opacity of this HistogramViewProperties.

        :return: The legend_opacity of this HistogramViewProperties.
        :rtype: float
        """  # noqa: E501
        return self._legend_opacity

    @legend_opacity.setter
    def legend_opacity(self, legend_opacity):
        """Set the legend_opacity of this HistogramViewProperties.

        :param legend_opacity: The legend_opacity of this HistogramViewProperties.
        :type: float
        """  # noqa: E501
        self._legend_opacity = legend_opacity

    @property
    def legend_orientation_threshold(self):
        """Get the legend_orientation_threshold of this HistogramViewProperties.

        :return: The legend_orientation_threshold of this HistogramViewProperties.
        :rtype: int
        """  # noqa: E501
        return self._legend_orientation_threshold

    @legend_orientation_threshold.setter
    def legend_orientation_threshold(self, legend_orientation_threshold):
        """Set the legend_orientation_threshold of this HistogramViewProperties.

        :param legend_orientation_threshold: The legend_orientation_threshold of this HistogramViewProperties.
        :type: int
        """  # noqa: E501
        self._legend_orientation_threshold = legend_orientation_threshold

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
        if not isinstance(other, HistogramViewProperties):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Return true if both objects are not equal."""
        return not self == other