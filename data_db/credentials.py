# uncompyle6 version 2.13.3
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.6 (default, Oct 26 2016, 20:30:19) 
# [GCC 4.8.4]
# Embedded file name: /home/drew/Documents/Allen_Brain_Observatory/db/credentials.py
# Compiled at: 2017-09-11 17:44:42


def postgresql_credentials():
    """Credentials for your postgres database."""
    return {'username': 'neural',
       'password': 'neural'  # on x9 it is neural
       }


def postgresql_connection(port=''):
    """Enter the name of your database below."""
    unpw = postgresql_credentials()
    params = {'database': 'neural',
       'user': unpw['username'],
       'password': unpw['password'],
       'host': 'localhost',
       'port': port
       }
    return params


def machine_credentials():
    """Credentials for your machine."""
    return {'username': 'drew',
       'password': 'serrelab',
       'ssh_address': 'serrep3.services.brown.edu'
       }
# okay decompiling credentials.pyc
