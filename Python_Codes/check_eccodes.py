import findlibs
path = findlibs.find("eccodes")
print("ecCodes library found at:", path)

import ecmwflibs
print("ecmwflibs location:", ecmwflibs.__file__)