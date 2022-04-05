# Add System Call

1. Add below line to `/linux/arch/x86/entry/syscalls/syscall_64.tbl`.
    ```
    600 common mycall sys_mycall
    ```
2. Add below line to `/linux/include/linux/syscalls.h`.
    ```c
    asmlinkage long sys_mycall(int num, char *message);
    ```
3. Create file `/linux/kernel/mycall.c`.
    ```c
    #include <linux/kernel.h>
    #include <linux/syscalls.h>
    #include <asm/uaccess.h>

    SYSCALL_DEFINE2(mycall, int, num, char *, message)
    {
        printk(" [ Kernel Message ] This message is from mycall system call with num=%d from user!\n", num);

        char msg[40];
        sprintf(msg, "Hello from kernel with %d!", num);

        copy_to_user(message, &msg, 40);

        return 0;
    }
    ```
4. Add `mycall.o` to `obj-y` in `/linux/kernel/Makefile`.
    ```
    obj-y = ...
            mycall.o
    ```
5. Build and Test!