def find_volumes(data_path):
    if data_path.is_dir():
        vol_volumes = data_path.glob("**/*.vol")
        xml_volumes = data_path.glob("**/*.xml")
        #eye_volumes = data_path.glob("**/*.eye")
        duke_volumes = data_path.glob("**/*.mat")
        # We do not support multiple XML exports in the same folder.
        xml_volumes = [v.parent for v in xml_volumes]
    elif data_path.is_file():
        xml_volumes = []
        vol_volumes = []
        #eye_volumes = []
        duke_volumes = []
        if ".vol" == data_path.suffix:
            vol_volumes.append(data_path)
        if ".xml" == data_path.suffix:
            xml_volumes.append(data_path)
        #if ".eye" == data_path.suffix:
        #    eye_volumes.append(data_path)
        if ".mat" == data_path.suffix:
            duke_volumes.append(data_path)

    else:
        raise ValueError("Data not found")

    return {"vol": set(vol_volumes), "xml": set(xml_volumes), "duke": set(duke_volumes)}#"eye": set(eye_volumes), }
