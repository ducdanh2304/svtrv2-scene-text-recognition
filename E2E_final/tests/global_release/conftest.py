"""The original SCOPE implementation uses flat imports; set its verified path for CPU tests."""
import sys
from str_e2e.backends.longtext import verify_scope_sources
from str_e2e.backends.official import SCOPE_SOURCE
verify_scope_sources()
sys.path.insert(0,str(SCOPE_SOURCE))
