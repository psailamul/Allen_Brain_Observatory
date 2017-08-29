
def postgresql_credentials():
    """Credentials for your postgres database."""
    return {
            'username': 'neural',
            'password': 'neural'
           }


def postgresql_connection(port=''):
    """Enter the name of your database below."""
    unpw = postgresql_credentials()
    params = {
        'database': 'neural',
        'user': unpw['username'],
        'password': unpw['password'],
        'host': 'localhost',
        'port': port,
    }
    return params


def machine_credentials():
    """Credentials for your machine."""
    return {
        'username': '',
        'password': '',
        'ssh_address': ''
       }
