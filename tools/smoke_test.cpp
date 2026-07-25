// smoke_test.cpp - valida que el toolchain y las deps compilan y enlazan
#include <cstdio>
#include <nlohmann/json.hpp>

int main() {
    nlohmann::json j;
    j["status"] = "ok";
    j["bits"] = sizeof(void*) * 8;
    printf("smoke_test: %s\n", j.dump().c_str());
    return 0;
}
