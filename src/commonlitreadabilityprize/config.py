from pathlib import Path


class Config:
    """ Container class for storing defaults """
    data = Path( Path().cwd() / "datasets" / "commonlitreadabilityprize" )
    seed = int(123)

if __name__ == "__main__":
    print( Config.data)