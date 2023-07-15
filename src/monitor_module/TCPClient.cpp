#include "TCPClient.h"

    TCPClient::TCPClient(const std::string address, const int port) : m_address(address), m_port(port)
    {
        m_isClose = false;
        std::cout << "TCPClient init success" << std::endl;
    }
    void TCPClient::runServer()
    {
        boost::asio::ip::tcp::endpoint EP(boost::asio::ip::address::from_string(m_address), m_port);
        m_TCPSocket = std::make_unique<boost::asio::ip::tcp::socket>(m_service);
        m_TCPSocket->async_connect(EP, boost::bind(&TCPClient::onConnect, this, _1));
        m_service.run();
    }
    void TCPClient::stopServer()
    {
        m_isClose = true;
        m_TCPSocket->cancel();
        m_service.stop();
    }
    void TCPClient::sendData(const std::ostringstream &msg)
    {
        {
            boost::shared_lock<boost::shared_mutex> writeLock(m_dataLock);
            m_dataLoader.push(msg.str());
        }
    }
    void TCPClient::sendDataThread()
    {
        while (1)
        {
            int m_isSend = 0;
            if (m_isClose == true)
            {
                break;
            }
            std::string tmpData;
            {
                if (m_dataLoader.size() > 0)
                {
                    tmpData = m_dataLoader.front();
                    m_dataLoader.pop();
                    m_isSend = 1;
                }
            }
            if (m_isSend == 1)
            {
                char sendBuffer[250] = {0};
                std::copy(tmpData.begin(), tmpData.end(), sendBuffer);
                m_TCPSocket->async_send(boost::asio::buffer(sendBuffer, tmpData.size()), boost::bind(&TCPClient::onSend, this, _1, _2));
            }
            boost::this_thread::sleep(boost::posix_time::milliseconds(1));
        }
    }
    void TCPClient::onConnect(const boost::system::error_code &err)
    {
        if (err)
        {
            std::cout << "建立TCP客户端连接失败！" << std::endl;
        }
        else
        {
            std::cout << "建立TCP客户端连接成功！" << std::endl;
        }
    }
    void TCPClient::onSend(const boost::system::error_code &err, size_t bytes)
    {
        if (!err)
        {
            std::cout << "向TCP服务端发送数据失败！" << std::endl;
        }
    }
    void TCPClient::readData(){

    }
    void TCPClient::onRead(const boost::system::error_code & err,size_t bytes){

    }
