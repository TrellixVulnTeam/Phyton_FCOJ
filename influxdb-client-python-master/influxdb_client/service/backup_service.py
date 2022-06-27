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


class BackupService(object):
    """NOTE: This class is auto generated by OpenAPI Generator.

    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):  # noqa: E501,D401,D403
        """BackupService - a operation defined in OpenAPI."""
        if api_client is None:
            raise ValueError("Invalid value for `api_client`, must be defined.")
        self.api_client = api_client

    def get_backup_kv(self, **kwargs):  # noqa: E501,D401,D403
        """Download snapshot of metadata stored in the server's embedded KV store. Should not be used in versions greater than 2.1.x, as it doesn't include metadata stored in embedded SQL..

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_backup_kv(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str zap_trace_span: OpenTracing span context
        :return: file
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_backup_kv_with_http_info(**kwargs)  # noqa: E501
        else:
            (data) = self.get_backup_kv_with_http_info(**kwargs)  # noqa: E501
            return data

    def get_backup_kv_with_http_info(self, **kwargs):  # noqa: E501,D401,D403
        """Download snapshot of metadata stored in the server's embedded KV store. Should not be used in versions greater than 2.1.x, as it doesn't include metadata stored in embedded SQL..

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_backup_kv_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str zap_trace_span: OpenTracing span context
        :return: file
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        local_var_params = locals()

        all_params = ['zap_trace_span']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')
        all_params.append('urlopen_kw')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_backup_kv" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}
        if 'zap_trace_span' in local_var_params:
            header_params['Zap-Trace-Span'] = local_var_params['zap_trace_span']  # noqa: E501

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/octet-stream', 'application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        # urlopen optional setting
        urlopen_kw = None
        if 'urlopen_kw' in kwargs:
            urlopen_kw = kwargs['urlopen_kw']

        return self.api_client.call_api(
            '/api/v2/backup/kv', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='file',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats,
            urlopen_kw=urlopen_kw)

    def get_backup_metadata(self, **kwargs):  # noqa: E501,D401,D403
        """Download snapshot of all metadata in the server.

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_backup_metadata(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str zap_trace_span: OpenTracing span context
        :param str accept_encoding: Indicates the content encoding (usually a compression algorithm) that the client can understand.
        :return: MetadataBackup
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_backup_metadata_with_http_info(**kwargs)  # noqa: E501
        else:
            (data) = self.get_backup_metadata_with_http_info(**kwargs)  # noqa: E501
            return data

    def get_backup_metadata_with_http_info(self, **kwargs):  # noqa: E501,D401,D403
        """Download snapshot of all metadata in the server.

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_backup_metadata_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str zap_trace_span: OpenTracing span context
        :param str accept_encoding: Indicates the content encoding (usually a compression algorithm) that the client can understand.
        :return: MetadataBackup
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        local_var_params = locals()

        all_params = ['zap_trace_span', 'accept_encoding']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')
        all_params.append('urlopen_kw')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_backup_metadata" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']

        collection_formats = {}

        path_params = {}

        query_params = []

        header_params = {}
        if 'zap_trace_span' in local_var_params:
            header_params['Zap-Trace-Span'] = local_var_params['zap_trace_span']  # noqa: E501
        if 'accept_encoding' in local_var_params:
            header_params['Accept-Encoding'] = local_var_params['accept_encoding']  # noqa: E501

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['multipart/mixed', 'application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        # urlopen optional setting
        urlopen_kw = None
        if 'urlopen_kw' in kwargs:
            urlopen_kw = kwargs['urlopen_kw']

        return self.api_client.call_api(
            '/api/v2/backup/metadata', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='MetadataBackup',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats,
            urlopen_kw=urlopen_kw)

    def get_backup_shard_id(self, shard_id, **kwargs):  # noqa: E501,D401,D403
        """Download snapshot of all TSM data in a shard.

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_backup_shard_id(shard_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int shard_id: The shard ID. (required)
        :param str zap_trace_span: OpenTracing span context
        :param str accept_encoding: Indicates the content encoding (usually a compression algorithm) that the client can understand.
        :param datetime since: Earliest time to include in the snapshot. RFC3339 format.
        :return: file
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_backup_shard_id_with_http_info(shard_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_backup_shard_id_with_http_info(shard_id, **kwargs)  # noqa: E501
            return data

    def get_backup_shard_id_with_http_info(self, shard_id, **kwargs):  # noqa: E501,D401,D403
        """Download snapshot of all TSM data in a shard.

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_backup_shard_id_with_http_info(shard_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int shard_id: The shard ID. (required)
        :param str zap_trace_span: OpenTracing span context
        :param str accept_encoding: Indicates the content encoding (usually a compression algorithm) that the client can understand.
        :param datetime since: Earliest time to include in the snapshot. RFC3339 format.
        :return: file
                 If the method is called asynchronously,
                 returns the request thread.
        """  # noqa: E501
        local_var_params = locals()

        all_params = ['shard_id', 'zap_trace_span', 'accept_encoding', 'since']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')
        all_params.append('urlopen_kw')

        for key, val in six.iteritems(local_var_params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_backup_shard_id" % key
                )
            local_var_params[key] = val
        del local_var_params['kwargs']
        # verify the required parameter 'shard_id' is set
        if ('shard_id' not in local_var_params or
                local_var_params['shard_id'] is None):
            raise ValueError("Missing the required parameter `shard_id` when calling `get_backup_shard_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'shard_id' in local_var_params:
            path_params['shardID'] = local_var_params['shard_id']  # noqa: E501

        query_params = []
        if 'since' in local_var_params:
            query_params.append(('since', local_var_params['since']))  # noqa: E501

        header_params = {}
        if 'zap_trace_span' in local_var_params:
            header_params['Zap-Trace-Span'] = local_var_params['zap_trace_span']  # noqa: E501
        if 'accept_encoding' in local_var_params:
            header_params['Accept-Encoding'] = local_var_params['accept_encoding']  # noqa: E501

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/octet-stream', 'application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = []  # noqa: E501

        # urlopen optional setting
        urlopen_kw = None
        if 'urlopen_kw' in kwargs:
            urlopen_kw = kwargs['urlopen_kw']

        return self.api_client.call_api(
            '/api/v2/backup/shards/{shardID}', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='file',  # noqa: E501
            auth_settings=auth_settings,
            async_req=local_var_params.get('async_req'),
            _return_http_data_only=local_var_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=local_var_params.get('_preload_content', True),
            _request_timeout=local_var_params.get('_request_timeout'),
            collection_formats=collection_formats,
            urlopen_kw=urlopen_kw)
