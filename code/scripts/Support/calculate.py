def get_coefficients(point_one: list[float],
                     point_two: list[float]):

    point_one = [float(item) for item in point_one]
    point_two = [float(item) for item in point_two]
    a = (point_two[1]-point_one[1])/(point_two[0]-point_one[0])
    b = (point_two[1] - a * point_two[0])

    return a, b
