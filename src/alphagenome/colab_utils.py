# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for Google Colab."""

import os
import pathlib


def _load_env_file(env_file_path: str = 'config.env') -> None:
  """Load environment variables from a .env file.
  
  Args:
    env_file_path: Path to the .env file relative to the project root.
  """
  # Find the project root (where config.env should be located)
  current_dir = pathlib.Path(__file__).parent
  project_root = current_dir
  
  # Walk up until we find the project root (contains pyproject.toml)
  while project_root != project_root.parent:
    if (project_root / 'pyproject.toml').exists():
      break
    project_root = project_root.parent
  
  env_file = project_root / env_file_path
  
  if env_file.exists():
    with open(env_file, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
          key, value = line.split('=', 1)
          # Only set if not already in environment
          if key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def get_api_key(secret: str = 'ALPHA_GENOME_API_KEY'):
  """Returns API key from environment variable or Colab secrets.

  Tries to retrieve the API key from the environment first. If not found,
  attempts to load from config.env file, then from Colab secrets (if running in Colab).

  Args:
    secret: The name of the environment variable or Colab secret key to
      retrieve.

  Raises:
    ValueError: If the API key cannot be found in the environment, config.env,
      or Colab secrets.
  """
  
  # First try to load from config.env if not already in environment
  _load_env_file()

  if api_key := os.environ.get(secret):
    return api_key

  try:
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    from google.colab import userdata  # pytype: disable=import-error
    # pylint: enable=g-import-not-at-top, import-outside-toplevel

    try:
      api_key = userdata.get(secret)
      return api_key
    except (
        userdata.NotebookAccessError,
        userdata.SecretNotFoundError,
        userdata.TimeoutException,
    ) as e:
      raise ValueError(
          f'Cannot find or access API key in Colab secrets with {secret=}. Make'
          ' sure you have added the API key to Colab secrets and enabled'
          ' access. See'
          ' https://www.alphagenomedocs.com/installation.html#add-api-key-to-secrets'
          ' for more details.'
      ) from e
  except ImportError:
    # Not running in Colab.
    pass

  raise ValueError(f'Cannot find API key with {secret=}.')
