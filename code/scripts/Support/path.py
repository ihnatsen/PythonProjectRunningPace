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


def get_path_dataset():
    return os.path.join(__root_project, 'dataset')


def get_path_to_new_file(path: list[str] | str, name_file: str):
    match path:
        case str():
            return os.path.join(__root_project, path, name_file)
        case list():
            return os.path.join(__root_project, *path, name_file)


__all__ = ['get_path_df', 'get_path_dataset', 'get_path_to_new_file']
