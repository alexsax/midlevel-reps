def pause_game(game):
    for i in range(1):
        r = game.make_action([False, False, False])

def spawn_agent(game, x, y, orientation=0):
    x_pos, y_pos = to_acs_float(x), to_acs_float(y)
    game.send_game_command("pukename set_position %i %i %i" %
                           (x_pos, y_pos, orientation))
    # pause_game(game)

def spawn_object(game, object_id, x, y, idx):
    # x_pos, y_pos = get_doom_coordinates(x, y)
    x_pos, y_pos = to_acs_float(x), to_acs_float(y)
    # call spawn function twice because vizdoom objects are not spawned
    # sometimes if spawned only once for some unknown reason
    for _ in range(1):
        game.send_game_command("pukename spawn_object_by_id_and_location1 \
                                %i %i %i" % (object_id, x_pos, y_pos))
        # game.send_game_command("pukename spawn_object_by_location %i %i" % (x_pos, y_pos))
        # print("pukename spawn_object_by_location %i %i" % (x_pos, y_pos))
        # game.send_game_command("pukename spawn_object_by_id %i" % (object_id, ))
        pause_game(game)

def to_acs_float(val):
    ''' ACS (doom script) has fixed-point numbers where the first 16 bits are the integer and
        the subsequent 16 bits are the decimal.'''
    return int(val * 2**16)

def to_python_float(val):
    return float(val) / 2**16










def make_torch_box(
        game,
        box_size_x=384,
        box_size_y=384,
        box_offset_x=0,
        box_offset_y=320,
        min_margin=30,
        n_torches_per_side=5,
        start_idx=0):
    ''' Makes an axis-aligned box of torches '''
    k = start_idx
    excess_x = (box_size_x - 2 * min_margin) % (n_torches_per_side-1)
    excess_y = (box_size_x - 2 * min_margin) % (n_torches_per_side-1)
    margin_x = int(min_margin + excess_x / 2)
    margin_y = int(min_margin + excess_y / 2)
    for i, x in enumerate(range(margin_x + box_offset_x,
                                box_size_x + box_offset_x - margin_x + 1,
                                int((box_size_x - 2 * margin_x) / (n_torches_per_side-1) ))):
        for j, y in enumerate(range(margin_y + box_offset_y,
                                    box_size_y + box_offset_y - margin_y + 1,
                                    int((box_size_y - 2 * margin_y) / (n_torches_per_side-1) ))):
            if i in [0, n_torches_per_side-1] and j in [0, n_torches_per_side-1]:
                spawn_object(game, 2, x, y, k)
            elif i in [0, n_torches_per_side-1] or j in [0, n_torches_per_side-1]:
                spawn_object(game, k, x, y, k)
            else:
                pass
            k += 1
