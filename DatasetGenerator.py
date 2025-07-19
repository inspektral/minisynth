import numpy as np
import ModGenerator
import MiniSynth

from Scale import Scale

class DatasetGenerator:

    def __init__(self, synth:MiniSynth):
        self.synth = synth
        self.sr = synth.sr
        self.duration = synth.duration
        self.samples = int(synth.sr * synth.duration)

        self.params = synth.get_parameters()

    def generate_modulations(self, num_active_mods:int = 1):
        if num_active_mods < 0 or num_active_mods > len(self.params.keys()):
            raise ValueError(f"Number of active mods must be between 0 and {len(self.params.keys())}")
        
        mod_generator = ModGenerator.ModGenerator()
        
        modulations = {}

        active_mods = np.random.choice(
            list(self.params.keys()), 
            size=num_active_mods, 
            replace=False
        )

        for mod in list(self.params.keys()):
            if mod in active_mods:
                minmax_array = np.sort(np.random.uniform(self.params[mod]["range"][0], self.params[mod]["range"][1], size=2))
            else:
                minmax_array = np.tile(np.random.uniform(self.params[mod]["range"][0], self.params[mod]["range"][1]), 2)

            modulations[mod] = mod_generator.init_points_random(
                min=minmax_array[0], 
                max=minmax_array[1], 
                scale=self.params[mod]["scale"]
            ).get_points()


        return modulations