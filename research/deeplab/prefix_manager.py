import os
from functools import wraps


def handle_valohai_prefix(environ):
    path = environ["PATH_INFO"]
    for prefix in (
        environ.get("HTTP_X_VH_PREFIX"),
        os.environ.get("VH_DEFAULT_PREFIX"),
    ):
        if not prefix:  # Could have no header or no envvar, so skip
            continue
        if path.startswith(prefix):  # If the path starts with this prefix,
            # ... then strip the prefix out as far as WSGI is concerned.
            environ["PATH_INFO"] = "/" + path[len(prefix) :].lstrip("/")
            break


def manage_prefixes(app):
    """
    Decorator to apply Valohai prefix management to a WSGI app callable.
    """

    @wraps(app)
    def prefix_managed_app(environ, start_response):
        handle_valohai_prefix(environ)
        return app(environ, start_response)

    return prefix_managed_app