/* fake_shm.c */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int shm_open(const char *name, int oflag, mode_t mode) {
    char path[256];
    const char *clean_name = (name && name[0] == '/') ? name + 1 : name;
    if (clean_name == NULL || clean_name[0] == '\0') {
        clean_name = "psm_default";
    }
    snprintf(path, sizeof(path), "/tmp/%s", clean_name);
    return open(path, oflag, mode);
}

int shm_unlink(const char *name) {
    char path[256];
    const char *clean_name = (name && name[0] == '/') ? name + 1 : name;
    if (clean_name == NULL || clean_name[0] == '\0') {
        clean_name = "psm_default";
    }
    snprintf(path, sizeof(path), "/tmp/%s", clean_name);
    return unlink(path);
}
