from grid_maps.maps import Map
from grid_maps.map_creator import MapCreator
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", help = "Name of the file where the map will be saved")
    parser.add_argument("--height", help = "Height of the map")
    parser.add_argument("--width", help = "Width of the map")

    args = parser.parse_args()

    map_creator = MapCreator(args.file)
    map_creator.render()