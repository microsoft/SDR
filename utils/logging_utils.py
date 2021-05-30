class Unbuffered(object):
    """
    Create buffer that dumps stdout to file.
    
    Example: sys.stdout = Unbuffered(open(path + '_output','w'))
    """

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)
