import numpy as np
import ModGenerator

ranges = {
    "base_freq": (20.0, 100.0),
    "amp": (0.0, 1.0),
    "mod_freq_ratio": (0.01, 10.0),
    "mod_shape": (0.0, 1.0),
    "carr_shape": (0.0, 1.0),
    "fm_amount": (0.0, 1.0)
}

def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def generate_modulations(num_active_mods:int = 1):
    if num_active_mods < 0 or num_active_mods > 6:
        raise ValueError("Number of active mods must be between 0 and 6.")
    
    mod_generator = ModGenerator.ModGenerator()
    
    modulations = {}

    active_mods = np.random.choice(
        list(ranges.keys()), 
        size=num_active_mods, 
        replace=False
    )

    for mod in list(ranges.keys()):
        if mod in active_mods:
            minmax_array = np.sort(np.random.uniform(ranges[mod][0], ranges[mod][1], size=2))
        else:
            minmax_array = np.tile(np.random.uniform(ranges[mod][0], ranges[mod][1]), 2)

        if mod == "base_freq":
            minmax_array = midi_to_freq(minmax_array)
            modulations[mod] = mod_generator.init_points_random(
                min=minmax_array[0], 
                max=minmax_array[1], 
                scale=ModGenerator.ModScale.LOGARITHMIC
            ).get_points()
        else:
            modulations[mod] = mod_generator.init_points_random(
                min=minmax_array[0], 
                max=minmax_array[1], 
                scale=ModGenerator.ModScale.LINEAR
            ).get_points()

    return modulations