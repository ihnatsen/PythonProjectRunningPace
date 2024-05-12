import os

__root_project = os.getcwd()
while not os.path.basename(__root_project) == 'code':
    __root_project = os.path.split(__root_project)[0]


def get_path_df(names_directory: list | str, name_df: str):
    match names_directory:
        case str():
            return os.path.join(__root_project, 'dataset', names_directory, name_df)
        case list():
            return os.path.join(__root_project, 'dataset', *names_directory, name_df)
        case _:
            raise ValueError('Path is not valid')

