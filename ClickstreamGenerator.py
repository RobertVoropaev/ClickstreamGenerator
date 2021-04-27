import pandas as pd
import numpy as np
import random
import datetime
import string
import argparse
import inspect
import logging
from tqdm import tqdm


class Clickstream_Generator:
    """
        Clickstream generator
    
        Parameters
        -------------
        max_user_num : int, default=100  
            The maximum number of users to be used in clickstream generation.
            
        max_pages : int, default=100
            The maximum number of pages to be used in clickstream generation.
            
        random_state: int, default=None
            Set random state for generation.

        pages_path : str, default="pages.txt"
            Local path to the file with the list of pages line by line.
    """
    
    def __init__(self, 
                 max_user_num: int = 100, 
                 max_pages: int = 100,
                 random_state: int = None,
                 pages_path : str = "pages.txt"):
        
        self.logger = logging.getLogger()
        self.logger.info(f"__init__: max_user_num {max_user_num}, max_pages {max_pages}, pages_path {pages_path}, random_state {random_state}")
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        self.clickstream = pd.DataFrame(columns=["user_id", "session_id", 
                                                 "event_type", "event_page", "timestamp"])
        self.lines = 0 
        self.bar = None
        
        self.max_user_num = max_user_num
        self.pages = self.load_pages(pages_path, max_pages)
        
        # variables of the current session
        self.users = [] 
        self.timestamp = int(datetime.datetime.now().timestamp()) 
    
    
    def load_pages(self, pages_path: str, max_pages: int):
        self.logger.info(f"load_pages: pages_path {pages_path}")
        
        pages = []
        with open(pages_path) as f:
            for line in f:
                pages.append(line.strip())
        
        p = np.random.randint(0, max_pages, size=max_pages)
        p_normed = p / p.sum()
        
        pages_dict = {}
        for i in range(max_pages):
            pages_dict[pages[i]] = p_normed[i]
        
        return pages_dict
     
    
    ############### Tools ###############
        
        
    def increase_timestamp(self):
        delta = random.randint(0, 10)
        self.logger.debug(f"increase_timestamp: {self.timestamp} -> {self.timestamp+delta}")
        self.timestamp += delta
    
    
    def get_last_action(self, user_id: int):
        last_action = self.clickstream[self.clickstream.user_id == user_id]        
        
        if last_action.shape[0]:
            self.logger.debug(f"get_last_action: user_id {user_id}, last_action: {last_action}")
            return last_action.sort_values("timestamp", ascending=False).iloc[0] 
        else:
            return None
        
    
    def append_in_clickstream(self, user_id: int, session_id : int, 
                              event_type: str, event_page: str):
        new_line = {"user_id": user_id, "session_id": session_id, 
                    "event_type": event_type, "event_page": event_page, 
                    "timestamp": self.timestamp}
        
        self.logger.debug(f"append_in_clickstream: new line {new_line}")
        self.clickstream = self.clickstream.append(new_line, ignore_index=True)
        
        self.lines += 1
        self.bar.update(1)
        

    def save_clickstream(self, file_path : str = "clickstream.csv"):
        self.logger.info(f"save_clickstream: file_path {file_path}")
        self.clickstream.to_csv(file_path, sep="\t", index=False)
    
    
    ############### Actions ###############
    
    
    def start_new_user_session(self):
        def get_new_user_id():
            user_id = random.randint(0, self.max_user_num)
            while user_id in self.users:
                user_id = random.randint(0, self.max_user_num)
            return user_id
        
        if len(self.users) >= self.max_user_num:
            return
        
        user_id = get_new_user_id()
        
        last_action = self.get_last_action(user_id)
        if last_action is not None:
            session_id = last_action.session_id + random.randint(0, 10)
        else:
            session_id = random.randint(0, 1000)
        
        self.logger.info(f"start_new_user_session: user_id {user_id} session_id {session_id}")
        
        self.users.append(user_id)
        self.append_in_clickstream(user_id=user_id, 
                                   session_id=session_id, 
                                   event_type="page", 
                                   event_page="main")
    
    
    def make_new_page_for_user(self, user_id: int):        
        def get_random_page(current_page: str):
            pages_sub = self.pages.copy()
            pages_sub.pop(current_page)
            p_sub = np.array(list(pages_sub.values()))
            page = np.random.choice(a=list(pages_sub.keys()), 
                                    size=1,
                                    p=p_sub / p_sub.sum())[0]
            return page
            
        
        last_action = self.get_last_action(user_id)
        new_page = get_random_page(current_page=last_action.event_page)

        self.logger.info(f"make_new_page_for_user: user_id {user_id}, new_page {new_page}")
        
        self.append_in_clickstream(user_id=user_id, 
                                   session_id=last_action.session_id, 
                                   event_type="page", 
                                   event_page=new_page)
        
        
    def make_event_for_user(self, user_id: int):
        self.logger.info(f"make_event_for_user: user_id {user_id}")
        
        last_action = self.get_last_action(user_id)
                
        self.append_in_clickstream(user_id=user_id, 
                                   session_id=last_action.session_id, 
                                   event_type="event", 
                                   event_page=last_action.event_page)
        
        
    def make_error_for_user(self, user_id: int):
        def generate_error_string():
            return "".join(random.choices(string.ascii_letters, 
                                          k=random.randint(0, 10))) + \
                   "error" + \
                   "".join(random.choices(string.ascii_letters, 
                                          k=random.randint(0, 10)))
        
        last_action = self.get_last_action(user_id)
        error_string = generate_error_string()
    
        self.logger.info(f"make_event_for_user: user_id {user_id}, error_string {error_string}")
        
        self.append_in_clickstream(user_id=user_id, 
                                   session_id=last_action.session_id, 
                                   event_type=error_string, 
                                   event_page=last_action.event_page)
        
        
    def stop_user_session(self, user_id: int):
        self.logger.info(f"stop_user_session: user_id {user_id}")
        self.users.remove(user_id)
                
     
    ############### Steps ###############
    
    
    def make_step(self, action_prob : dict):
        """
            Make a random action for a random user.
            
            Parameters
            -------------
            action_prob : dict  
                Dictionary of actions and their probabilities.
        """
        action = np.random.choice(a=list(action_prob.keys()), 
                                  size=1,
                                  p=list(action_prob.values()))[0]
        
        if "user_id" in inspect.signature(action).parameters:
            if self.users:
                user_id = random.choice(self.users)
                action(user_id)
            else:
                pass
        else:
            action()
        
        self.increase_timestamp()
    
    
    def run(self, 
            n_lines: int = 1000,
            start_user_session_prob : float = 0.05, 
            page_for_user_prob : float = 0.33,
            event_for_user_prob : float = 0.55,
            error_for_user_prob : float = 0.02,
            stop_user_session_prob : float = 0.05): 
        
        """
            Start the clickstream generation process

            Parameters
            -------------
            n_lines : int, default=1000  
                The number of lines in clickstream.

            start_user_session_prob : float, default=0.05
                The probability that a new user session will start.
            
            page_for_user_prob : float, default=0.33
                The probability that a user will go to a new page.
            
            event_for_user_prob : float, default=0.55
                The probability that a user will take an action on the current page.
            
            error_for_user_prob : float, default=0.02
                The probability that an error will occur on the current page.
            
            stop_user_session_prob : float, default=0.05
                The probability that a current user session will stop.
                
            verbose : int, default=None
                Period in epochs of outputting the iteration number.
        """
        
        action_prob = {
            self.start_new_user_session: start_user_session_prob, 
            self.make_new_page_for_user: page_for_user_prob, 
            self.make_event_for_user: event_for_user_prob, 
            self.make_error_for_user: error_for_user_prob, 
            self.stop_user_session: stop_user_session_prob
        }
        
        self.bar = tqdm(total=n_lines)
        while self.lines != n_lines:
            self.make_step(action_prob)
                
    

def run_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--n_lines", type=int, required=False, default=1000,
                        help="number of lines")
    
    parser.add_argument("-u", "--max_users", type=int, required=False, default=100,
                        help="max number of users")
    
    parser.add_argument("-start_p", "--start_user_session_prob", type=float, required=False, default=0.05,
                        help="probability that a new user session will start")

    parser.add_argument("-page_p", "--page_for_user_prob", type=float, required=False, default=0.33,
                        help="probability that a user will go to a new page")

    parser.add_argument("-event_p", "--event_for_user_prob", type=float, required=False, default=0.55,
                        help="probability that a user will take an action on the current page")
        
    parser.add_argument("-error_p", "--error_for_user_prob", type=float, required=False, default=0.02,
                        help="probability that an error will occur on the current page")

    parser.add_argument("-stop_p", "--stop_user_session_prob", type=float, required=False, default=0.05,
                        help="probability that a current user session will stop")

    
    args = parser.parse_args()
    
    return args

            
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    args = run_argparse()
    
    click_gen = Clickstream_Generator(max_user_num=args.max_users)
    
    click_gen.run(n_lines=args.n_lines,
                  start_user_session_prob=args.start_user_session_prob,
                  page_for_user_prob=args.page_for_user_prob,
                  event_for_user_prob=args.event_for_user_prob,
                  error_for_user_prob=args.error_for_user_prob,
                  stop_user_session_prob=args.stop_user_session_prob)
    
    click_gen.save_clickstream()