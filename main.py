from grid_maps.maps import Map
from grid_maps.map_creator import MapCreator
from argparse import ArgumentParser

from cv2 import imread, imshow, waitKey, destroyAllWindows

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", help = "Name of the file where the map will be saved")
    parser.add_argument("--height", help = "Height of the map", type = int)
    parser.add_argument("--width", help = "Width of the map", type = int)
    parser.add_argument("--max_timesteps", help = "Maximum timesteps for the map", type = int)

    args = parser.parse_args()

    map_creator = MapCreator(args.file, args.height, args.width)
    map_creator.render()

    map = Map(args.file, args.max_timesteps)
    observations, infos = map.reset()
    
    while map.agents:
        actions = {
            agent : map.action_space(agent).sample() for agent in map.agents
        }
        
        observations, rewards, terminations, trunctations, infos = map.step(actions)
    map.close()
    
    ts = map.timestep
    
    for i in range(map.timestep):
        img = imread(f"images/img_{i}.png")
        imshow("animation", img)
        
        if i == map.timestep - 1:
            k = waitKey(1000) & 0xff
        else:
            k = waitKey(100) & 0xff
        
        if k == 27:
            break
        
    destroyAllWindows()
        
