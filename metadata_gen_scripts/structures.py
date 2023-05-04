import copy

instrument_list = ["tru", "gac", "sax", "cel", "flu", "gel", "vio", "cla", "pia", "org", "voi"]
genre_list = ["cla", "jaz_blu", "pop_roc", "cou_fol", "lat_sou"]

instrument_dict = {
    "tru": 0,
    "gac": 0,
    "sax": 0,
    "cel": 0,
    "flu": 0,
    "gel": 0,
    "vio": 0,
    "cla": 0,
    "pia": 0,
    "org": 0,
    "voi": 0
}

instrument_mix_counter = {
    "tru": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "gac": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "sax": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "flu": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "gel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "vio": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cla": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "pia": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "org": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "voi": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

specific_instrument_counter = {
    "tru": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "gac": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "sax": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "cel": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "flu": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "gel": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "vio": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "cla": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "pia": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "org": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))],
    "voi": [copy.deepcopy(instrument_dict) for _ in range(len(instrument_list))]
}

num_of_companion_instruments_percentage = {
    "tru": [],
    "gac": [],
    "sax": [],
    "cel": [],
    "flu": [],
    "gel": [],
    "vio": [],
    "cla": [],
    "pia": [],
    "org": [],
    "voi": []
}

instrument_percentages_dict = {
    "tru": [[], [], [], [], [], [], [], [], [], [], []],
    "gac": [[], [], [], [], [], [], [], [], [], [], []],
    "sax": [[], [], [], [], [], [], [], [], [], [], []],
    "cel": [[], [], [], [], [], [], [], [], [], [], []],
    "flu": [[], [], [], [], [], [], [], [], [], [], []],
    "gel": [[], [], [], [], [], [], [], [], [], [], []],
    "vio": [[], [], [], [], [], [], [], [], [], [], []],
    "cla": [[], [], [], [], [], [], [], [], [], [], []],
    "pia": [[], [], [], [], [], [], [], [], [], [], []],
    "org": [[], [], [], [], [], [], [], [], [], [], []],
    "voi": [[], [], [], [], [], [], [], [], [], [], []]
}

