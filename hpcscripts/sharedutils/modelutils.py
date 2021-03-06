import os

def ListModels(models_directory: str):
    models_list = []
    dirs = os.listdir(models_directory)

    for item in dirs:
        item_path = os.path.join(models_directory, item)
        if os.path.isdir(item_path):
            models_list.append(item_path)

    return models_list

def SelectModelPrompt(models_directory, select: int=None):

    models_list = ListModels(models_directory)
    models_list.sort()

    print ("Found {} models inside {}:".format(
                                        len(models_list), 
                                        models_directory
                                    ))
    
    print ("index    Model-name")
    for i, models_path in enumerate(models_list):
        number = "[{}]. ".format(i).ljust(7, " ")
        print ("  {}{}".format(
                            number, 
                            os.path.basename (models_path)
                        ))
    if select == None:
        index = input("Please input your model's index (e.g 0): ")
        index = min(max(int(index), 0), len(models_list)-1)
    else:
        index = min(max(select, 0), len(models_list)-1) # clamp select

    print ("You selected model {}".format(os.path.basename (models_list[index])))

    return models_list[index]