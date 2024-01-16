import logging

try:
    # 配置基本的日志设置
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        handlers=[logging.FileHandler('logs/example.log'), logging.StreamHandler()]
                        )

    # 示例日志
    logging.debug('This is a debug message')
    logging.info('This is an info message')
    logging.warning('This is a warning message')
    logging.error('This is an error message')
    logging.critical('This is a critical message')

except Exception as e:
    print(f"An error occurred: {e}")
