import logging
import time

def timer(name: str = "Unnamed process"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info(f"[[{name}]] - Start Running...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logging.error(f"[[{name}]]: Execution Error")
                raise e
            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"[[{name}]] - Finished Running...")
            logging.info(f"[[{name}]] - Total time cost: {total_time:.4f} seconds")
            return result
        return wrapper
    return decorator
