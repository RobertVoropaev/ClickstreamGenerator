import pandas as pd
import numpy as np
import random
import datetime
import string
import argparse


class Clickstream_Generator:
    def __init__(self, max_user_count: int = 100):
        self.clickstream = pd.DataFrame(columns=["user_id", "session_id", 
                                                 "event_type", "event_page", "timestamp"])
        self.load_pages()
        
        self.current_users = []
        self.current_session = 1
        self.timestamp = int(datetime.datetime.now().timestamp())
        self.max_user_count = max_user_count
    
    
    def append_in_clickstream(self, user_id: int, session_id: int, 
                              event_type: str, event_page: str, timestamp: int):
        
        self.clickstream = self.clickstream.append({"user_id": user_id, "session_id": session_id, 
                                                    "event_type": event_type, "event_page": event_page, 
                                                    "timestamp": timestamp}, ignore_index=True)
        
    
    def load_pages(self):
        self.pages = []
        with open("pages.txt") as f:
            for line in f:
                self.pages.append(line.strip())
    
    
    def save_clickstream(self):
        self.clickstream.to_csv("clickstream.csv", sep="\t", index=False)
    
    
    def increase_timestamp(self):
        self.timestamp += random.randint(0, 10)
    
    
    def increase_current_session(self):
        print(f"increase_current_session {self.current_session} -> {self.current_session+1}")
        self.current_session += 1
    
    
    def get_last_action(self, user_id: int):
         return self.clickstream[self.clickstream.user_id == user_id] \
                    .sort_values("timestamp", ascending=False).iloc[0]
    
    
    def start_new_user_session(self):
        def get_new_user_id():
            user_id = random.randint(0, self.max_user_count)
            while user_id in self.current_users:
                user_id = random.randint(0, self.max_user_count)
            return user_id
        
        
        
        user_id = get_new_user_id()
        session_id = self.current_session
        timestamp = self.timestamp
        event_type = "page"
        event_page = "main"
        
        print(f"start_new_user_session {user_id}")
        self.current_users.append(user_id)
        self.append_in_clickstream(user_id, session_id, event_type, event_page, timestamp)
    
    
    def make_new_page_for_user(self, user_id: int):
        print(f"make_new_page_for_user {user_id}")
        def get_random_page(current: str):
            page_index = random.randint(0, len(self.pages) - 1)
            while page_index == current:
                page_index = random.randint(0, len(self.pages) - 1)
            return self.pages[page_index]
        
        last_action = self.get_last_action(user_id)
        
        session_id = last_action.session_id
        timestamp = self.timestamp
        event_type = "page"
        event_page = get_random_page(last_action.event_page)
        
        self.append_in_clickstream(user_id, session_id, event_type, event_page, timestamp)
        
        
    def make_event_for_user(self, user_id: int):
        print(f"make_event_for_user {user_id}")
        last_action = self.get_last_action(user_id)
        
        session_id = last_action.session_id
        timestamp = self.timestamp
        event_type = "event"
        event_page = last_action.event_page
        
        self.append_in_clickstream(user_id, session_id, event_type, event_page, timestamp)
        
        
    def make_error_for_user(self, user_id: int):
        print(f"make_error_for_user {user_id}")
        def get_error_string():
            n1 = random.randint(0, 10)
            n2 = random.randint(0, 10)
            return "".join(random.choices(string.ascii_letters, k=n1)) + \
                   "error" + \
                   "".join(random.choices(string.ascii_letters, k=n2))
        
        last_action = self.get_last_action(user_id)
        
        session_id = last_action.session_id
        timestamp = self.timestamp
        event_type = get_error_string()
        event_page = last_action.event_page
        
        self.append_in_clickstream(user_id, session_id, event_type, event_page, timestamp)
        
        
    def stop_user_session(self, user_id: int):
        print(f"stop_user_session {user_id}")
        self.current_users.remove(user_id)
                
        
    def step(self):
        def get_random_current_user():
            return random.choice(self.current_users + [None])
        
        user_id = get_random_current_user()
        
        if user_id is None:
            action_list = [self.start_new_user_session, self.increase_current_session]
            action = random.choice(action_list)
            action()
        else:
            action_list = [self.make_new_page_for_user, 
                           self.make_event_for_user, self.make_error_for_user, self.stop_user_session]

            action = random.choice(action_list)
            action(user_id)
        
        self.increase_timestamp()
    
    
    def make_steps(self, n: int):
        for i in range(0, n):
            self.step()
    

def run_argparse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--n_steps", type=int, required=True,
                        help="number of steps")
    
    parser.add_argument("-u", "--n_users", type=int, required=True,
                        help="max number of users")
    
    args = parser.parse_args()
    
    return args

            
if __name__ == "__main__":
    args = run_argparse()
    
    click_gen = Clickstream_Generator(args.n_users)
    click_gen.make_steps(args.n_steps)
    click_gen.save_clickstream()