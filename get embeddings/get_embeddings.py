# Get vectors for elements from your trained mat2vec model

import pandas as pd
from gensim.models import Word2Vec

# Loading the trained mat2vec model

w2v_model = Word2Vec.load("mat2vec/training/models/robo_description_final_model")

# Specifying the elements we want to extract embeddings for

elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
                "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
                "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
                "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Ac", "Th", "Pa", "U", "Np", "Pu"]

element_vecs = []

for e in elements:
    el_vector = w2v_model.wv[e]
    element_vecs.append(el_vector)
    
el_vec = pd.DataFrame(element_vecs)
name_vec = pd.DataFrame(elements)
element_vectors = pd.concat([name_vec, el_vec], axis=1)

# Save the vectors

element_vectors.to_csv('materials_properties/robo_descriptor.csv', index=True)