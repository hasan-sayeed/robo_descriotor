import os
import pandas as pd

from pymatgen.ext.matproj import MPRester
from robocrys import StructureCondenser, StructureDescriber
from os.path import join
from pathlib import Path

# query stable materials

props = ["task_id", "pretty_formula", "spacegroup"]

with MPRester(<your API key>) as m:
    results = m.query(
        {"e_above_hull": {"$lt": 0.5}},
        properties=props,
    )


# seperate mpids and fromulas
mpids = [d["task_id"] for d in results]
formulas = [d["pretty_formula"] for d in results]

# combine into DataFrame
df = pd.DataFrame({"formula": formulas, "task_id": mpids})
df.to_csv("data/mp_list.csv", index = False)

# Generating robocrystallographer descriptions

df = pd.read_csv('data/mp_list.csv')

data_dir = join('data', 'description')
data_folder = Path(data_dir).mkdir(parents=True, exist_ok=True)

condenser = StructureCondenser()
describer = StructureDescriber()

# Looping through the stable materials from materials project and generating description

for j in df["task_id"]:
    structure = MPRester(<your API key>).get_structure_by_material_id(j)
    
    condensed_structure = condenser.condense_structure(structure)
    description = describer.describe(condensed_structure)
    with open(os.path.join(data_dir,'robo_descriptions_corpus'), "a") as the_file:
        the_file.write(description + '\n')
    

the_file.close()