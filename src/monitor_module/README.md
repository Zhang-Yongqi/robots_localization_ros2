

```cpp
 std::ostringstream MonitorData;
        MonitorData << "Data:"
                    << (float)_armors[0].m_T.at<double>(0, 0) << ","
                    << (float)_armors[0].m_T.at<double>(1, 0) << ","
                    << (float)_armors[0].m_T.at<double>(2, 0) << ","
                    << (int)_armors[0].m_number << ","
                    << "\n";
        param::Monitor->sendData(MonitorData);
```
