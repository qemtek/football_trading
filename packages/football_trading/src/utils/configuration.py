import os

from src.utils.logging import get_logger
import configuration

logger = get_logger()


def get_attribute(attribute_name, fail_if_not_found=True):
    """Get credentials attribute required in the project. First
    check the environment variables, then the configuration.py file"""

    if os.environ.get(attribute_name) is None:
        logger.warning(f'{attribute_name} is not specified as an environment variable')
        if hasattr(configuration, attribute_name):
            logger.warning(f'Retrieving {attribute_name} from configuration.py file')
            return getattr(configuration, attribute_name)
        else:
            if fail_if_not_found:
                raise AttributeError(f'Cannot get {attribute_name} from environment or configuration.py file')
            else:
                logger.warning(f'Cannot get {attribute_name} from the environment or '
                               f'configuration.py file, returning None')
                return None
    else:
        return os.environ.get(attribute_name)
