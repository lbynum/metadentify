import os
from unittest.mock import MagicMock, patch

from metadentify.utils import get_available_cpus


def test_get_available_cpus_macos():
    with patch('os.environ', {}):
        with patch('os.cpu_count', return_value=8):
            with patch('os.path.exists', return_value=False):
                with patch('metadentify.utils.os', spec=os) as mock_os:
                    mock_os.environ = {}
                    mock_os.cpu_count.return_value = 8
                    delattr(mock_os, 'sched_getaffinity')
                    cpus = get_available_cpus()
                    assert cpus == 8


def test_get_available_cpus_slurm():
    with patch.dict('os.environ', {'SLURM_CPUS_ON_NODE': '12'}):
        cpus = get_available_cpus()
        assert cpus == 12


def test_get_available_cpus_linux_affinity():
    with patch('os.environ', {}):
        with patch('metadentify.utils.os', spec=os) as mock_os:
            mock_os.environ = {}
            mock_os.sched_getaffinity = MagicMock(return_value={0, 1, 2, 3})
            cpus = get_available_cpus()
            assert cpus == 4
