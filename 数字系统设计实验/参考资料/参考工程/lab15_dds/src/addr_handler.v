module addr_handler(raw_addr, rom_addr);
    // The module is for processing address

    input [10:0] raw_addr;
    output reg[9:0] rom_addr;

    always @(*) begin
        if (raw_addr[10])
            begin
                if (raw_addr == 1024)
                    rom_addr = 1023;
                else
                    rom_addr = ~raw_addr[9:0] + 1;
            end
        else
            rom_addr = raw_addr[9:0];
    end
endmodule
