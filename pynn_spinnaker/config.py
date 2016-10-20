import os
import os.path
import appdirs

from six import iteritems
from six.moves.configparser import ConfigParser, NoOptionError


# The application name to use in config file names
_name = "pynn_spinnaker_0_8"

# Standard config file names/locations
SYSTEM_CONFIG_FILE = appdirs.site_config_dir(_name)
USER_CONFIG_FILE = appdirs.user_config_dir(_name)
CWD_CONFIG_FILE = os.path.join(os.curdir, ".{}".format(_name))

# Search path for config files (lowest to highest priority)
SEARCH_PATH = [
    SYSTEM_CONFIG_FILE,
    USER_CONFIG_FILE,
    CWD_CONFIG_FILE,
]

def _add_section(parser, section_name, defaults):
    parser.add_section(section_name)
    for key, value in iteritems(defaults):
        parser.set(section_name, key,
                   "None" if value is None else str(value))

def read_config(filenames=SEARCH_PATH):
    """Attempt to read local configuration files to determine spalloc client
    settings.
    Parameters
    ----------
    filenames : [str, ...]
        Filenames to attempt to read. Later config file have higher priority.
    Returns
    -------
    dict
        The configuration loaded.
    """
    parser = ConfigParser()

    '''
    config.add_section("Mapping")
    config.add_section("Machine")

    # Mapping section
    config.set("Mapping", "extra_xmls_paths", "None")

    # Machine section
    config.setint("Machine", "appID", 66)

    '''
    # Set default config values (NB: No read_dict in Python 2.7)
    _add_section(parser, "Mapping",
                 {"extra_xmls_paths": None,
                  "machine_graph_to_machine_algorithms":
                      "RadialPlacer,RigRoute,BasicTagAllocator,"
                      "FrontEndCommonEdgeToNKeysMapper,"
                      "MallocBasedRoutingInfoAllocator,"
                      "BasicRoutingTableGenerator,MundyRouterCompressor"})
    _add_section(parser, "Machine",
                 {"appID": 66,
                  "virtual_board": False,
                  "turn_off_machine": False,
                  "clear_routing_tables": False,
                  "clear_tags": False,
                  "enable_reinjection": False,
                  "max_sdram_allowed_per_chip": None,
                  "bmp_names": None,
                  "down_chips": None,
                  "down_cores": None,
                  "down_links": None,
                  "auto_detect_bmp": False,
                  "scamp_connections_data": None,
                  "boot_connection_port_num": None,
                  "version": 5,
                  "reset_machine_on_startup": False,
                  "core_limit": None,
                  "post_simulation_overrun_before_error": 5,
                  "DSEAppID": 31,
                  "requires_wrap_arounds": None})
    _add_section(parser, "Reports",
                 {"defaultReportFilePath": "DEFAULT",
                  "max_reports_kept": 10,
                  "max_application_binaries_kept": 10,
                  "defaultapplicationdatafilepath": "DEFAULT",
                  "writealgorithmtimings": False,
                  "display_algorithm_timings": False,
                  "provenance_format": "xml",
                  "reportsEnabled": False,
                  "writeprovenancedata": False,
                  "extract_iobuf": False,
                  "writetextspecs": False,
                  "writeMemoryMapReport": False})
    _add_section(parser, "SpecExecution",
                 {"specExecOnHost": True })
    _add_section(parser, "Database",
                 {"create_database": False,
                  "wait_on_confirmation": True,
                  "send_start_notification": True})
    _add_section(parser, "Mode",
                 {"verify_writes": False})
    _add_section(parser, "Buffers",
                 {"use_auto_pause_and_resume": False})

    # Attempt to read from each possible file location in turn
    for filename in filenames:
        try:
            with open(filename, "r") as f:
                parser.readfp(f, filename)
        except (IOError, OSError):
            # File did not exist, keep trying
            pass

    return parser