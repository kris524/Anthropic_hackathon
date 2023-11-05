def create_text(text_filepath: str) -> str:
    # given a filepath to a text file, return the text as a string removing newlines
    with open(text_filepath, "r") as file:
        text = file.read().replace("\n", " ")
    return text
