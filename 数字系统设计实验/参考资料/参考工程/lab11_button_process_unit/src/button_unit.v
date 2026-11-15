module button_unit(clk, reset, ButtonIn, ButtonOut);
    parameter sim = 0; // the default value of sim is 0
    input clk, reset, ButtonIn;
    output ButtonOut;

    wire sync_to_deb, deb_to_pwc;
    synchronizer sync_inst(.in(ButtonIn), .clk(clk), .out(sync_to_deb));
    debouncer #(.sim(sim)) deb_inst(.reset(reset), .in(sync_to_deb), .clk(clk), .out(deb_to_pwc));
    pulse_width_change pwc_inst(.in(deb_to_pwc), .clk(clk), .out(ButtonOut));

endmodule
