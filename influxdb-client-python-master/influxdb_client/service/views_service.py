# coding: utf-8

"""
InfluxDB OSS API Service.

The InfluxDB v2 API provides a programmatic interface for all interactions with InfluxDB. Access the InfluxDB API using the `/api/v2/` endpoint.   # noqa: E501

OpenAPI spec version: 2.0.0
Generated by: https://openapi-generator.tech
"""


from __future__ import absolute_import

import re  # noqa: F401

# python 2 and python 3 compatibility library
import six


class ViewsService(object):
    """NOTE: This class is auto generated by OpenAPI Generator.

    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):  # noqa: E501,D401,D403
        """ViewsService - a operation defined in OpenAPI."""
        if api_client is None:
            raise ValueError("Invalid value for `api_client`, must be defined.")
        self.api_client = api_client

    def get_dashboards_id_cells_id_view(self, dashboard_id, cell_id, **kwargs):  # noqa: E501,D401,D403
        """Retrieve the view for a cell.

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_dashboards_id_cells_id_view(dashboard_id, cell_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str dashboard_id: The dashboard ID. (required)
        :param str cell_id: The cell ID. (required)
        :param str zap_trace_span: OpenTracing span context
        :return: View
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_dashboards_id_cells_id_view_with_http_info(dashboard_id, cell_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_dashboards_id_cells_id_view_with_http_info(dashboard_id, cell_id, **kwargs)  # noqa: E501
            return data

    def get_dashboards_id_cells_id_view_with_http_info(self, dashboard_id, cell_id, **kwargs):  # noqa: E501,D401,D403
        """Retrieve the view for a cell.

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_dashboards_id_cells_id_view_with_http_info(dashboard_id, cell_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str dashboard_id: The dashboard ID. (required)
        :param str cell_id: The cell ID. (required)
        :param str zap_trace_span: OpenTracing span context
        :return: View
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        local_var_params = locals()

        all_params = ['dashboard_id', 'cell_id', 'zap_trace_span']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')
        all_params.append('urlopen_kw')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_dashboards_id_cells_id_view" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']
        # verify the required parameter 'dashboard_id' is set
        if ('dashboard_id' not in local_var_params or
                local_var_params['dashboard_id'] is None):
            raise ValueError("Missing the required parameter `dashboard_id` when calling `get_dashboards_id_cells_id_view`")  # noqa: E501
        # verify the required parameter 'cell_id' is set
        if ('cell_id' not in local_var_params or
                local_var_params['cell_id'] is None):
            raise ValueError("Missing the required parameter `cell_id` when calling `get_dashboards_id_cells_id_view`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dashboard_id' in local_var_params:
            path_params['dashboardID'] = local_var_params['dashboard_id']  # noqa: E501
        if 'cell_id' in local_var_params:
            path_params['cellID'] = local_var_params['cell_id']  # noqa: E501

        query_params = []

        header_params = {}
        if 'zap_trace_span' in local_var_params:
            header_params['Zap-Trace-Span'] = local_var_params['zap_trace_span']  # noqa: E501

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        # urlopen optional setting
        urlopen_kw = None
        if 'urlopen_kw' in kwargs:
            urlopen_kw = kwargs['urlopen_kw']

        return self.api_client.call_api(
            '/api/v2/dashboards/{dashboardID}/cells/{cellID}/view', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='View',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats,
            urlopen_kw=urlopen_kw)

    def patch_dashboards_id_cells_id_view(self, dashboard_id, cell_id, view, **kwargs):  # noqa: E501,D401,D403
        """Update the view for a cell.

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.patch_dashboards_id_cells_id_view(dashboard_id, cell_id, view, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str dashboard_id: The ID of the dashboard to update. (required)
        :param str cell_id: The ID of the cell to update. (required)
        :param View view: (required)
        :param str zap_trace_span: OpenTracing span context
        :return: View
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.patch_dashboards_id_cells_id_view_with_http_info(dashboard_id, cell_id, view, **kwargs)  # noqa: E501
        else:
            (data) = self.patch_dashboards_id_cells_id_view_with_http_info(dashboard_id, cell_id, view, **kwargs)  # noqa: E501
            return data

    def patch_dashboards_id_cells_id_view_with_http_info(self, dashboard_id, cell_id, view, **kwargs):  # noqa: E501,D401,D403
        """Update the view for a cell.

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.patch_dashboards_id_cells_id_view_with_http_info(dashboard_id, cell_id, view, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str dashboard_id: The ID of the dashboard to update. (required)
        :param str cell_id: The ID of the cell to update. (required)
        :param View view: (required)
        :param str zap_trace_span: OpenTracing span context
        :return: View
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        local_var_params = locals()

        all_params = ['dashboard_id', 'cell_id', 'view', 'zap_trace_span']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')
        all_params.append('urlopen_kw')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method patch_dashboards_id_cells_id_view" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']
        # verify the required parameter 'dashboard_id' is set
        if ('dashboard_id' not in local_var_params or
                local_var_params['dashboard_id'] is None):
            raise ValueError("Missing the required parameter `dashboard_id` when calling `patch_dashboards_id_cells_id_view`")  # noqa: E501
        # verify the required parameter 'cell_id' is set
        if ('cell_id' not in local_var_params or
                local_var_params['cell_id'] is None):
            raise ValueError("Missing the required parameter `cell_id` when calling `patch_dashboards_id_cells_id_view`")  # noqa: E501
        # verify the required parameter 'view' is set
        if ('view' not in local_var_params or
                local_var_params['view'] is None):
            raise ValueError("Missing the required parameter `view` when calling `patch_dashboards_id_cells_id_view`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dashboard_id' in local_var_params:
            path_params['dashboardID'] = local_var_params['dashboard_id']  # noqa: E501
        if 'cell_id' in local_var_params:
            path_params['cellID'] = local_var_params['cell_id']  # noqa: E501

        query_params = []

        header_params = {}
        if 'zap_trace_span' in local_var_params:
            header_params['Zap-Trace-Span'] = local_var_params['zap_trace_span']  # noqa: E501

        form_params = []
        local_var_files = {}

        body_params = None
        if 'view' in local_var_params:
            body_params = local_var_params['view']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        # urlopen optional setting
        urlopen_kw = None
        if 'urlopen_kw' in kwargs:
            urlopen_kw = kwargs['urlopen_kw']

        return self.api_client.call_api(
            '/api/v2/dashboards/{dashboardID}/cells/{cellID}/view', 'PATCH',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='View',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats,
            urlopen_kw=urlopen_kw)