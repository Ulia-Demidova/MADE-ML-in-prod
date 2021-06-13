import pickle


def save_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(path: str) -> object:
    with open(path, 'rb') as f:
        model = pickle.loads(f.read())
    return model
