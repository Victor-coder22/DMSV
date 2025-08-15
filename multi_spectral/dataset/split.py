from itertools import groupby

def get_recordingenv(msi_item):
    k, v = msi_item
    return v['path'].split('/')[4]

def by_value(id):
    def __by__(item):
        return item[1][id]
    return __by__


def train_val(msi_dict, split=""):

    train_dict = {}
    val_dict = {}
    
    grouped_by_folder = groupby(msi_dict.items(), get_recordingenv,)
    for folder, msi_data in grouped_by_folder:
        msi_data = dict(msi_data)
        sorted_msi_data = sorted(msi_data.items(), key=by_value('path'))
        sorted_ids = [msi_item[0] for msi_item in sorted(msi_data.items(), key=by_value('path'))]
        if type(split) is dict:
            val_idx = split[folder]
            val_dict.update(
                dict([sorted_msi_data[idx] for idx in val_idx])
            )
            train_dict.update(
                dict([sorted_msi_data[idx] for idx in range(len(sorted_msi_data)) if idx not in val_idx])
            )
        elif split == "5th":
            val_dict.update(
                {id:msi_data[id] for id in sorted_ids[2::5]}
            )
            train_dict.update(
                {id:msi_data[id] for id in sorted_ids if id not in sorted_ids[2::5]}
            )
        elif split == "end":
            train_dict.update(
                {id:msi_data[id] for id in sorted_ids[:int(len(sorted_ids) * 0.81)]}
                #.8 leads to 10 images less in trian compared to every 5th
            ) 
            val_dict.update(
                {id:msi_data[id] for id in sorted_ids[int(len(sorted_ids) * 0.81):]}
            )
        else:
            #TODO throw error
            return
    
    return train_dict, val_dict