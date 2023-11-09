class Config(object):
    DEBUG = False
    TESTING = False
    # any other default config properties

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
