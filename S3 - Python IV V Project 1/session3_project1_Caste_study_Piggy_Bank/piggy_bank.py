# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 20:39:29 2018

@author: manoj
"""
class InvalidEntry(Exception):pass

class piggy_bank:
    def __init__(self,balance=0.0):
        self.balance=balance
        self.bonus_points=0
    def add_amount(self,amount):
        self.balance = self.balance+amount
        if amount >= 100:
            self.bonus_points = self.bonus_points + int(amount//100)
        print("After adding, your balance is {} rupees and bonus points is {}".format(self.balance,self.bonus_points))
        self.is_zero_balance()
    def withdraw_amount(self,amount):
        if amount > self.balance:
            self.check_balance()
            print("Please enter an amount which is less than or equal to the balance amount.")
        else:
            self.balance = self.balance-amount
            print("After withdrawing, your balance is {} rupees".format(self.balance))
            self.is_zero_balance()
    def check_balance(self):
        print("Your current balance is {} rupees and bonus points is {}".format(self.balance,self.bonus_points))
        self.is_zero_balance()
    def is_zero_balance(self):
        if self.balance==0:
            print("If you have money in your savings account you will get benefits like interest and bonus points.")
    def start(self):
        keep_running=True
        while keep_running:
            print('')
            print("Start or End : ",end="")
            start_stop=input()
            try:
                if start_stop.lower()=="start":
                    print("Add, Withdraw or Check : ",end="")
                    next_action=input()
                    if next_action.lower()=="add":
                        print("Add amount (Note:- for every 100 rupees you will get 1 bonus points): ",end="")
                        amount=int(input())
                        self.add_amount(amount)
                    elif next_action.lower()=="withdraw":
                        print("Withdraw amount : ",end="")
                        amount=int(input())
                        self.withdraw_amount(amount)
                    elif next_action.lower()=="check":
                        self.check_balance()
                    else:
                        raise InvalidEntry()
                elif start_stop.lower()=="end":
                    keep_running=False
                else:
                    raise InvalidEntry
            except ValueError:
                print("Please enter a valid integer")
            except InvalidEntry:
                print("Please enter a valid option")
            finally:
                print("")