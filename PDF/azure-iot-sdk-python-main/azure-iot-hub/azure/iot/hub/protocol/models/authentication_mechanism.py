# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from msrest.serialization import Model


class AuthenticationMechanism(Model):
    """AuthenticationMechanism.

    :param symmetric_key: The primary and secondary keys used for SAS based
     authentication.
    :type symmetric_key: ~protocol.models.SymmetricKey
    :param x509_thumbprint: The primary and secondary x509 thumbprints used
     for x509 based authentication.
    :type x509_thumbprint: ~protocol.models.X509Thumbprint
    :param type: The type of authentication used to connect to the service.
     Possible values include: 'sas', 'selfSigned', 'certificateAuthority',
     'none'
    :type type: str or ~protocol.models.enum
    """

    _attribute_map = {
        "symmetric_key": {"key": "symmetricKey", "type": "SymmetricKey"},
        "x509_thumbprint": {"key": "x509Thumbprint", "type": "X509Thumbprint"},
        "type": {"key": "type", "type": "str"},
    }

    def __init__(self, **kwargs):
        super(AuthenticationMechanism, self).__init__(**kwargs)
        self.symmetric_key = kwargs.get("symmetric_key", None)
        self.x509_thumbprint = kwargs.get("x509_thumbprint", None)
        self.type = kwargs.get("type", None)