from cv2 import imread

class Map:
    def __init__(self, file_name):
        self.map = imread(file_name)
        self.array = Map.prepare(self.map)

    @staticmethod
    def prepare(map):
        assert len(map.shape) == 3, "Map should be rgb image"
        array = map.mean(axis = 2)
        return array