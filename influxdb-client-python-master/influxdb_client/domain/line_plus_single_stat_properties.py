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


class LinePlusSingleStatProperties(ViewProperties):
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
        'time_format': 'str',
        'type': 'str',
        'queries': 'list[DashboardQuery]',
        'colors': 'list[DashboardColor]',
        'shape': 'str',
        'note': 'str',
        'show_note_when_empty': 'bool',
        'axes': 'Axes',
        'static_legend': 'StaticLegend',
        'x_column': 'str',
        'generate_x_axis_ticks': 'list[str]',
        'x_total_ticks': 'int',
        'x_tick_start': 'float',
        'x_tick_step': 'float',
        'y_column': 'str',
        'generate_y_axis_ticks': 'list[str]',
        'y_total_ticks': 'int',
        'y_tick_start': 'float',
        'y_tick_step': 'float',
        'shade_below': 'bool',
        'hover_dimension': 'str',
        'position': 'str',
        'prefix': 'str',
        'suffix': 'str',
        'decimal_places': 'DecimalPlaces',
        'legend_colorize_rows': 'bool',
        'legend_hide': 'bool',
        'legend_opacity': 'float',
        'legend_orientation_threshold': 'int'
    }

    attribute_map = {
        'time_format': 'timeFormat',
        'type': 'type',
        'queries': 'queries',
        'colors': 'colors',
        'shape': 'shape',
        'note': 'note',
        'show_note_when_empty': 'showNoteWhenEmpty',
        'axes': 'axes',
        'static_legend': 'staticLegend',
        'x_column': 'xColumn',
        'generate_x_axis_ticks': 'generateXAxisTicks',
        'x_total_ticks': 'xTotalTicks',
        'x_tick_start': 'xTickStart',
        'x_tick_step': 'xTickStep',
        'y_column': 'yColumn',
        'generate_y_axis_ticks': 'generateYAxisTicks',
        'y_total_ticks': 'yTotalTicks',
        'y_tick_start': 'yTickStart',
        'y_tick_step': 'yTickStep',
        'shade_below': 'shadeBelow',
        'hover_dimension': 'hoverDimension',
        'position': 'position',
        'prefix': 'prefix',
        'suffix': 'suffix',
        'decimal_places': 'decimalPlaces',
        'legend_colorize_rows': 'legendColorizeRows',
        'legend_hide': 'legendHide',
        'legend_opacity': 'legendOpacity',
        'legend_orientation_threshold': 'legendOrientationThreshold'
    }

    def __init__(self, time_format=None, type=None, queries=None, colors=None, shape=None, note=None, show_note_when_empty=None, axes=None, static_legend=None, x_column=None, generate_x_axis_ticks=None, x_total_ticks=None, x_tick_start=None, x_tick_step=None, y_column=None, generate_y_axis_ticks=None, y_total_ticks=None, y_tick_start=None, y_tick_step=None, shade_below=None, hover_dimension=None, position=None, prefix=None, suffix=None, decimal_places=None, legend_colorize_rows=None, legend_hide=None, legend_opacity=None, legend_orientation_threshold=None):  # noqa: E501,D401,D403
        """LinePlusSingleStatProperties - a model defined in OpenAPI."""  # noqa: E501
        ViewProperties.__init__(self)  # noqa: E501

        self._time_format = None
        self._type = None
        self._queries = None
        self._colors = None
        self._shape = None
        self._note = None
        self._show_note_when_empty = None
        self._axes = None
        self._static_legend = None
        self._x_column = None
        self._generate_x_axis_ticks = None
        self._x_total_ticks = None
        self._x_tick_start = None
        self._x_tick_step = None
        self._y_column = None
        self._generate_y_axis_ticks = None
        self._y_total_ticks = None
        self._y_tick_start = None
        self._y_tick_step = None
        self._shade_below = None
        self._hover_dimension = None
        self._position = None
        self._prefix = None
        self._suffix = None
        self._decimal_places = None
        self._legend_colorize_rows = None
        self._legend_hide = None
        self._legend_opacity = None
        self._legend_orientation_threshold = None
        self.discriminator = None

        if time_format is not None:
            self.time_format = time_format
        self.type = type
        self.queries = queries
        self.colors = colors
        self.shape = shape
        self.note = note
        self.show_note_when_empty = show_note_when_empty
        self.axes = axes
        if static_legend is not None:
            self.static_legend = static_legend
        if x_column is not None:
            self.x_column = x_column
        if generate_x_axis_ticks is not None:
            self.generate_x_axis_ticks = generate_x_axis_ticks
        if x_total_ticks is not None:
            self.x_total_ticks = x_total_ticks
        if x_tick_start is not None:
            self.x_tick_start = x_tick_start
        if x_tick_step is not None:
            self.x_tick_step = x_tick_step
        if y_column is not None:
            self.y_column = y_column
        if generate_y_axis_ticks is not None:
            self.generate_y_axis_ticks = generate_y_axis_ticks
        if y_total_ticks is not None:
            self.y_total_ticks = y_total_ticks
        if y_tick_start is not None:
            self.y_tick_start = y_tick_start
        if y_tick_step is not None:
            self.y_tick_step = y_tick_step
        if shade_below is not None:
            self.shade_below = shade_below
        if hover_dimension is not None:
            self.hover_dimension = hover_dimension
        self.position = position
        self.prefix = prefix
        self.suffix = suffix
        self.decimal_places = decimal_places
        if legend_colorize_rows is not None:
            self.legend_colorize_rows = legend_colorize_rows
        if legend_hide is not None:
            self.legend_hide = legend_hide
        if legend_opacity is not None:
            self.legend_opacity = legend_opacity
        if legend_orientation_threshold is not None:
            self.legend_orientation_threshold = legend_orientation_threshold

    @property
    def time_format(self):
        """Get the time_format of this LinePlusSingleStatProperties.

        :return: The time_format of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._time_format

    @time_format.setter
    def time_format(self, time_format):
        """Set the time_format of this LinePlusSingleStatProperties.

        :param time_format: The time_format of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        self._time_format = time_format

    @property
    def type(self):
        """Get the type of this LinePlusSingleStatProperties.

        :return: The type of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._type

    @type.setter
    def type(self, type):
        """Set the type of this LinePlusSingleStatProperties.

        :param type: The type of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")  # noqa: E501
        self._type = type

    @property
    def queries(self):
        """Get the queries of this LinePlusSingleStatProperties.

        :return: The queries of this LinePlusSingleStatProperties.
        :rtype: list[DashboardQuery]
        """  # noqa: E501
        return self._queries

    @queries.setter
    def queries(self, queries):
        """Set the queries of this LinePlusSingleStatProperties.

        :param queries: The queries of this LinePlusSingleStatProperties.
        :type: list[DashboardQuery]
        """  # noqa: E501
        if queries is None:
            raise ValueError("Invalid value for `queries`, must not be `None`")  # noqa: E501
        self._queries = queries

    @property
    def colors(self):
        """Get the colors of this LinePlusSingleStatProperties.

        Colors define color encoding of data into a visualization

        :return: The colors of this LinePlusSingleStatProperties.
        :rtype: list[DashboardColor]
        """  # noqa: E501
        return self._colors

    @colors.setter
    def colors(self, colors):
        """Set the colors of this LinePlusSingleStatProperties.

        Colors define color encoding of data into a visualization

        :param colors: The colors of this LinePlusSingleStatProperties.
        :type: list[DashboardColor]
        """  # noqa: E501
        if colors is None:
            raise ValueError("Invalid value for `colors`, must not be `None`")  # noqa: E501
        self._colors = colors

    @property
    def shape(self):
        """Get the shape of this LinePlusSingleStatProperties.

        :return: The shape of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._shape

    @shape.setter
    def shape(self, shape):
        """Set the shape of this LinePlusSingleStatProperties.

        :param shape: The shape of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        if shape is None:
            raise ValueError("Invalid value for `shape`, must not be `None`")  # noqa: E501
        self._shape = shape

    @property
    def note(self):
        """Get the note of this LinePlusSingleStatProperties.

        :return: The note of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._note

    @note.setter
    def note(self, note):
        """Set the note of this LinePlusSingleStatProperties.

        :param note: The note of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        if note is None:
            raise ValueError("Invalid value for `note`, must not be `None`")  # noqa: E501
        self._note = note

    @property
    def show_note_when_empty(self):
        """Get the show_note_when_empty of this LinePlusSingleStatProperties.

        If true, will display note when empty

        :return: The show_note_when_empty of this LinePlusSingleStatProperties.
        :rtype: bool
        """  # noqa: E501
        return self._show_note_when_empty

    @show_note_when_empty.setter
    def show_note_when_empty(self, show_note_when_empty):
        """Set the show_note_when_empty of this LinePlusSingleStatProperties.

        If true, will display note when empty

        :param show_note_when_empty: The show_note_when_empty of this LinePlusSingleStatProperties.
        :type: bool
        """  # noqa: E501
        if show_note_when_empty is None:
            raise ValueError("Invalid value for `show_note_when_empty`, must not be `None`")  # noqa: E501
        self._show_note_when_empty = show_note_when_empty

    @property
    def axes(self):
        """Get the axes of this LinePlusSingleStatProperties.

        :return: The axes of this LinePlusSingleStatProperties.
        :rtype: Axes
        """  # noqa: E501
        return self._axes

    @axes.setter
    def axes(self, axes):
        """Set the axes of this LinePlusSingleStatProperties.

        :param axes: The axes of this LinePlusSingleStatProperties.
        :type: Axes
        """  # noqa: E501
        if axes is None:
            raise ValueError("Invalid value for `axes`, must not be `None`")  # noqa: E501
        self._axes = axes

    @property
    def static_legend(self):
        """Get the static_legend of this LinePlusSingleStatProperties.

        :return: The static_legend of this LinePlusSingleStatProperties.
        :rtype: StaticLegend
        """  # noqa: E501
        return self._static_legend

    @static_legend.setter
    def static_legend(self, static_legend):
        """Set the static_legend of this LinePlusSingleStatProperties.

        :param static_legend: The static_legend of this LinePlusSingleStatProperties.
        :type: StaticLegend
        """  # noqa: E501
        self._static_legend = static_legend

    @property
    def x_column(self):
        """Get the x_column of this LinePlusSingleStatProperties.

        :return: The x_column of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._x_column

    @x_column.setter
    def x_column(self, x_column):
        """Set the x_column of this LinePlusSingleStatProperties.

        :param x_column: The x_column of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        self._x_column = x_column

    @property
    def generate_x_axis_ticks(self):
        """Get the generate_x_axis_ticks of this LinePlusSingleStatProperties.

        :return: The generate_x_axis_ticks of this LinePlusSingleStatProperties.
        :rtype: list[str]
        """  # noqa: E501
        return self._generate_x_axis_ticks

    @generate_x_axis_ticks.setter
    def generate_x_axis_ticks(self, generate_x_axis_ticks):
        """Set the generate_x_axis_ticks of this LinePlusSingleStatProperties.

        :param generate_x_axis_ticks: The generate_x_axis_ticks of this LinePlusSingleStatProperties.
        :type: list[str]
        """  # noqa: E501
        self._generate_x_axis_ticks = generate_x_axis_ticks

    @property
    def x_total_ticks(self):
        """Get the x_total_ticks of this LinePlusSingleStatProperties.

        :return: The x_total_ticks of this LinePlusSingleStatProperties.
        :rtype: int
        """  # noqa: E501
        return self._x_total_ticks

    @x_total_ticks.setter
    def x_total_ticks(self, x_total_ticks):
        """Set the x_total_ticks of this LinePlusSingleStatProperties.

        :param x_total_ticks: The x_total_ticks of this LinePlusSingleStatProperties.
        :type: int
        """  # noqa: E501
        self._x_total_ticks = x_total_ticks

    @property
    def x_tick_start(self):
        """Get the x_tick_start of this LinePlusSingleStatProperties.

        :return: The x_tick_start of this LinePlusSingleStatProperties.
        :rtype: float
        """  # noqa: E501
        return self._x_tick_start

    @x_tick_start.setter
    def x_tick_start(self, x_tick_start):
        """Set the x_tick_start of this LinePlusSingleStatProperties.

        :param x_tick_start: The x_tick_start of this LinePlusSingleStatProperties.
        :type: float
        """  # noqa: E501
        self._x_tick_start = x_tick_start

    @property
    def x_tick_step(self):
        """Get the x_tick_step of this LinePlusSingleStatProperties.

        :return: The x_tick_step of this LinePlusSingleStatProperties.
        :rtype: float
        """  # noqa: E501
        return self._x_tick_step

    @x_tick_step.setter
    def x_tick_step(self, x_tick_step):
        """Set the x_tick_step of this LinePlusSingleStatProperties.

        :param x_tick_step: The x_tick_step of this LinePlusSingleStatProperties.
        :type: float
        """  # noqa: E501
        self._x_tick_step = x_tick_step

    @property
    def y_column(self):
        """Get the y_column of this LinePlusSingleStatProperties.

        :return: The y_column of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._y_column

    @y_column.setter
    def y_column(self, y_column):
        """Set the y_column of this LinePlusSingleStatProperties.

        :param y_column: The y_column of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        self._y_column = y_column

    @property
    def generate_y_axis_ticks(self):
        """Get the generate_y_axis_ticks of this LinePlusSingleStatProperties.

        :return: The generate_y_axis_ticks of this LinePlusSingleStatProperties.
        :rtype: list[str]
        """  # noqa: E501
        return self._generate_y_axis_ticks

    @generate_y_axis_ticks.setter
    def generate_y_axis_ticks(self, generate_y_axis_ticks):
        """Set the generate_y_axis_ticks of this LinePlusSingleStatProperties.

        :param generate_y_axis_ticks: The generate_y_axis_ticks of this LinePlusSingleStatProperties.
        :type: list[str]
        """  # noqa: E501
        self._generate_y_axis_ticks = generate_y_axis_ticks

    @property
    def y_total_ticks(self):
        """Get the y_total_ticks of this LinePlusSingleStatProperties.

        :return: The y_total_ticks of this LinePlusSingleStatProperties.
        :rtype: int
        """  # noqa: E501
        return self._y_total_ticks

    @y_total_ticks.setter
    def y_total_ticks(self, y_total_ticks):
        """Set the y_total_ticks of this LinePlusSingleStatProperties.

        :param y_total_ticks: The y_total_ticks of this LinePlusSingleStatProperties.
        :type: int
        """  # noqa: E501
        self._y_total_ticks = y_total_ticks

    @property
    def y_tick_start(self):
        """Get the y_tick_start of this LinePlusSingleStatProperties.

        :return: The y_tick_start of this LinePlusSingleStatProperties.
        :rtype: float
        """  # noqa: E501
        return self._y_tick_start

    @y_tick_start.setter
    def y_tick_start(self, y_tick_start):
        """Set the y_tick_start of this LinePlusSingleStatProperties.

        :param y_tick_start: The y_tick_start of this LinePlusSingleStatProperties.
        :type: float
        """  # noqa: E501
        self._y_tick_start = y_tick_start

    @property
    def y_tick_step(self):
        """Get the y_tick_step of this LinePlusSingleStatProperties.

        :return: The y_tick_step of this LinePlusSingleStatProperties.
        :rtype: float
        """  # noqa: E501
        return self._y_tick_step

    @y_tick_step.setter
    def y_tick_step(self, y_tick_step):
        """Set the y_tick_step of this LinePlusSingleStatProperties.

        :param y_tick_step: The y_tick_step of this LinePlusSingleStatProperties.
        :type: float
        """  # noqa: E501
        self._y_tick_step = y_tick_step

    @property
    def shade_below(self):
        """Get the shade_below of this LinePlusSingleStatProperties.

        :return: The shade_below of this LinePlusSingleStatProperties.
        :rtype: bool
        """  # noqa: E501
        return self._shade_below

    @shade_below.setter
    def shade_below(self, shade_below):
        """Set the shade_below of this LinePlusSingleStatProperties.

        :param shade_below: The shade_below of this LinePlusSingleStatProperties.
        :type: bool
        """  # noqa: E501
        self._shade_below = shade_below

    @property
    def hover_dimension(self):
        """Get the hover_dimension of this LinePlusSingleStatProperties.

        :return: The hover_dimension of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._hover_dimension

    @hover_dimension.setter
    def hover_dimension(self, hover_dimension):
        """Set the hover_dimension of this LinePlusSingleStatProperties.

        :param hover_dimension: The hover_dimension of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        self._hover_dimension = hover_dimension

    @property
    def position(self):
        """Get the position of this LinePlusSingleStatProperties.

        :return: The position of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._position

    @position.setter
    def position(self, position):
        """Set the position of this LinePlusSingleStatProperties.

        :param position: The position of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        if position is None:
            raise ValueError("Invalid value for `position`, must not be `None`")  # noqa: E501
        self._position = position

    @property
    def prefix(self):
        """Get the prefix of this LinePlusSingleStatProperties.

        :return: The prefix of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._prefix

    @prefix.setter
    def prefix(self, prefix):
        """Set the prefix of this LinePlusSingleStatProperties.

        :param prefix: The prefix of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        if prefix is None:
            raise ValueError("Invalid value for `prefix`, must not be `None`")  # noqa: E501
        self._prefix = prefix

    @property
    def suffix(self):
        """Get the suffix of this LinePlusSingleStatProperties.

        :return: The suffix of this LinePlusSingleStatProperties.
        :rtype: str
        """  # noqa: E501
        return self._suffix

    @suffix.setter
    def suffix(self, suffix):
        """Set the suffix of this LinePlusSingleStatProperties.

        :param suffix: The suffix of this LinePlusSingleStatProperties.
        :type: str
        """  # noqa: E501
        if suffix is None:
            raise ValueError("Invalid value for `suffix`, must not be `None`")  # noqa: E501
        self._suffix = suffix

    @property
    def decimal_places(self):
        """Get the decimal_places of this LinePlusSingleStatProperties.

        :return: The decimal_places of this LinePlusSingleStatProperties.
        :rtype: DecimalPlaces
        """  # noqa: E501
        return self._decimal_places

    @decimal_places.setter
    def decimal_places(self, decimal_places):
        """Set the decimal_places of this LinePlusSingleStatProperties.

        :param decimal_places: The decimal_places of this LinePlusSingleStatProperties.
        :type: DecimalPlaces
        """  # noqa: E501
        if decimal_places is None:
            raise ValueError("Invalid value for `decimal_places`, must not be `None`")  # noqa: E501
        self._decimal_places = decimal_places

    @property
    def legend_colorize_rows(self):
        """Get the legend_colorize_rows of this LinePlusSingleStatProperties.

        :return: The legend_colorize_rows of this LinePlusSingleStatProperties.
        :rtype: bool
        """  # noqa: E501
        return self._legend_colorize_rows

    @legend_colorize_rows.setter
    def legend_colorize_rows(self, legend_colorize_rows):
        """Set the legend_colorize_rows of this LinePlusSingleStatProperties.

        :param legend_colorize_rows: The legend_colorize_rows of this LinePlusSingleStatProperties.
        :type: bool
        """  # noqa: E501
        self._legend_colorize_rows = legend_colorize_rows

    @property
    def legend_hide(self):
        """Get the legend_hide of this LinePlusSingleStatProperties.

        :return: The legend_hide of this LinePlusSingleStatProperties.
        :rtype: bool
        """  # noqa: E501
        return self._legend_hide

    @legend_hide.setter
    def legend_hide(self, legend_hide):
        """Set the legend_hide of this LinePlusSingleStatProperties.

        :param legend_hide: The legend_hide of this LinePlusSingleStatProperties.
        :type: bool
        """  # noqa: E501
        self._legend_hide = legend_hide

    @property
    def legend_opacity(self):
        """Get the legend_opacity of this LinePlusSingleStatProperties.

        :return: The legend_opacity of this LinePlusSingleStatProperties.
        :rtype: float
        """  # noqa: E501
        return self._legend_opacity

    @legend_opacity.setter
    def legend_opacity(self, legend_opacity):
        """Set the legend_opacity of this LinePlusSingleStatProperties.

        :param legend_opacity: The legend_opacity of this LinePlusSingleStatProperties.
        :type: float
        """  # noqa: E501
        self._legend_opacity = legend_opacity

    @property
    def legend_orientation_threshold(self):
        """Get the legend_orientation_threshold of this LinePlusSingleStatProperties.

        :return: The legend_orientation_threshold of this LinePlusSingleStatProperties.
        :rtype: int
        """  # noqa: E501
        return self._legend_orientation_threshold

    @legend_orientation_threshold.setter
    def legend_orientation_threshold(self, legend_orientation_threshold):
        """Set the legend_orientation_threshold of this LinePlusSingleStatProperties.

        :param legend_orientation_threshold: The legend_orientation_threshold of this LinePlusSingleStatProperties.
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
        if not isinstance(other, LinePlusSingleStatProperties):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Return true if both objects are not equal."""
        return not self == other