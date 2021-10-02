#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 14:20:56 2021

@author: gajender
"""

import httplib2
url="https://api.telegram.org/bot2027042757:AAGvZ-S4vuPpvXsSffElV498sjjWGorosuw/sendmessage?chat_id=-448906214&text="
message = "test_message"
httplib2.Http().request(url+message)
