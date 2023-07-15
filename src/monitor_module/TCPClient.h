#pragma once

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <chrono>
#include <boost/bind.hpp>
#include <iostream>
#include <queue>

class TCPClient
    {
    public:
        TCPClient(const std::string address, const int port);
        void runServer();
        void stopServer();
        void sendData(const std::ostringstream &msg);
        void sendDataThread();
        void readData();

    private:
        void onConnect(const boost::system::error_code & err);
        void onSend(const boost::system::error_code & err,size_t bytes);
        void onRead(const boost::system::error_code & err,size_t bytes);
        const int m_port;
        const std::string m_address;
        boost::asio::io_service m_service;
        std::shared_ptr<boost::asio::ip::tcp::socket> m_TCPSocket;
        boost::shared_mutex m_dataLock;
        std::queue<std::string> m_dataLoader;
        bool m_isClose;
    };
