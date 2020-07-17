#include <algorithm>
#include <iostream>
#include <regex>
#include <string>

#include "httplib.h"
using ll = long long;

std::string modulate(ll x) {
    std::string res = "";
    {  // signal
        if (x >= 0)
            res += "01";
        else
            res += "10";
    }
    x           = abs(x);
    int bit_len = 0;
    {  // bit length
        while ((1ll << bit_len) <= x) bit_len++;
        bit_len = ((bit_len / 4) + (bit_len % 4 ? 1 : 0)) * 4;
        for (int i = 0; i < bit_len / 4; i++) res += "1";
        res += "0";
    }
    {  // number
        std::string res2 = "";
        while (x > 0) {
            res2 += x % 2 ? "1" : "0";
            x /= 2;
        }
        while (res2.size() < bit_len) res2 += "0";
        std::reverse(std::begin(res2), std::end(res2));
        res += res2;
    }
    return res;
}

void test_modulate() {
    assert(modulate(0) == "010");
    assert(modulate(1) == "01100001");
    assert(modulate(-1) == "10100001");
    assert(modulate(2) == "01100010");
    assert(modulate(-2) == "10100010");
    assert(modulate(16) == "0111000010000");
    assert(modulate(-16) == "1011000010000");
    assert(modulate(255) == "0111011111111");
    assert(modulate(-255) == "1011011111111");
    assert(modulate(256) == "011110000100000000");
    assert(modulate(-256) == "101110000100000000");
}

int main(int argc, char* argv[]) {
    test_modulate();
    const std::string serverUrl(argv[1]);
    const std::string playerKey(argv[2]);

    std::cout << "ServerUrl: " << serverUrl << "; PlayerKey: " << playerKey
              << std::endl;

    const std::regex urlRegexp("http://(.+):(\\d+)");
    std::smatch urlMatches;
    if (!std::regex_search(serverUrl, urlMatches, urlRegexp) ||
        urlMatches.size() != 3) {
        std::cout << "Unexpected server response:\nBad server URL" << std::endl;
        return 1;
    }
    const std::string serverName = urlMatches[1];
    const int serverPort         = std::stoi(urlMatches[2]);
    httplib::Client client(serverName, serverPort);

    ll x0                 = std::stoi(playerKey);
    std::string ap_mod_x0 = modulate(x0);

    const std::shared_ptr<httplib::Response> serverResponse =
        client.Post(serverUrl.c_str(), ap_mod_x0, "text/plain");

    if (!serverResponse) {
        std::cout << "Unexpected server response:\nNo response from server"
                  << std::endl;
        return 1;
    }

    if (serverResponse->status != 200) {
        std::cout << "Unexpected server response:\nHTTP code: "
                  << serverResponse->status
                  << "\nResponse body: " << serverResponse->body << std::endl;
        return 2;
    }

    std::cout << "Server response: " << serverResponse->body << std::endl;
    return 0;
}
