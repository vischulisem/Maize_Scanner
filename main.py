#!/usr/bins/python3

import pandas as pd
import xml.etree.ElementTree as et

xtree = et.parse("/Users/elysevischulis/Downloads/X401x492-2m1.xml")
xroot = xtree.getroot()

df_cols = ["Type", "MarkerX", "MarkerY"]
out_df = pd.DataFrame(columns=df_cols)

for node in xroot:
    type = node.attrib.get("Type")
    markerx = node.find("MarkerX").text if node is not None else None
    markery = node.find("MarkerY").text if node is not None else None


    out_df = out_df.append(pd.Series([type, markerx, markery],
                                     index=df_cols),
                           ignore_index=True)

print(out_df)