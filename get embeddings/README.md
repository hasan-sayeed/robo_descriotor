* `mpid_to_robo_descriptions.py` queries materials project for stable materials and leverage robocrystallographer to generate crystallographic descriptions for those materials.

* To train mat2vec, clone [mat2vec repo](https://github.com/materialsintelligence/mat2vec) and keep it in the main directory of this repository. To train mat2vec on the generated robocrystallographer description corpus navigate to `mat2vec/training` and run like following:

```python phrase2vec.py --corpus=data/description/robo_descriptions_corpus --model_name=put_a_model_name_here```

* This will train mat2vec on your provided corpus and save the trained model in `mat2vec/training/models` directory. Running `get_embeddings.py` will give you embedding vectors for the elements of periodic table.