from abc import ABC

import gym


class RLFPSv4(gym.Env, ABC):
    def __init__(self, env_config):
        self.__version = 4
        self.__envConfig = env_config
        self.__pause_mode = self.__envConfig['mode'] == "Pause"
        self.__period = self.__envConfig['period']
        self.__communication_manager = CommunicationManager(self.__envConfig['port'])

    def initialize(self):
        self.__communication_manager.send_info(
            'Initialize:NoM"' + str(self.__envConfig['map_capacity']) + '"NoT"' + str(
                int(self.__envConfig['agent_count'] / self.__envConfig['max_n_of_friend']))
            + '"Pause"' + str(self.__pause_mode) + '"')

    def step(self, action: list):
        # Send action list
        action_buffer = self.__communication_manager.serialize_action_list(action)
        self.__communication_manager.send_info(action_buffer)

        self.__communication_manager.send_info("Break")
        if self.__pause_mode:
            time.sleep(self.__period / 1000)
        self.__communication_manager.send_info("Pause")

        # non-spatial, spatial, step info 받아오기
        return self.__get_observation()

    def reset(self):
        self_play = self.__envConfig['self_play']
        n_of_current_agent = self.__envConfig['max_n_of_friend']
        n_of_current_enemy = self.__envConfig['max_n_of_enemy']
        msg = 'Rebuild:"Selfplay"' + str(self_play) + \
                  '"NoA"' + str(n_of_current_agent) + \
                  '"NoE"' + str(n_of_current_enemy) + \
                  '"Parent"' + str(self.__envConfig['parent_name']) + \
                  '"Base"' + str(self.__envConfig['base_name']) + \
                  '"GlRe"' + str(self.__envConfig['global_resolution']) + \
                  '"LcRe"' + str(self.__envConfig['local_resolution']) + \
                  '"InitHeight"' + str(self.__envConfig['init_height']) + '"'
        self.__communication_manager.send_info(msg)
        # To gathering spatial-feature
        self.__communication_manager.send_info("Break")
        self.__communication_manager.send_info("Pause")
        return self.__get_observation()

    def __get_observation(self):
        character_info_list = self.__communication_manager.send_info(
            'CharacterInfo: "' + str(self.__envConfig['non_spatial_dim']) + '"')
        field_info_list = self.__communication_manager.send_info(
            'FieldInfo: "' + str(self.__envConfig['spatial_dim']) + '"')
        step_results = self.__communication_manager.send_info('Step: "' + str(self.__envConfig['step_info_dim']) + '"')

        deserialized_char_info_list = self.__communication_manager.deserialize_info_list(character_info_list)
        agent_num = self.__envConfig['max_n_of_friend']
        batch_num = self.__envConfig['spatial_dim']
        deserialized_field_info_list = self.__communication_manager.deserialize_field_info_list(field_info_list,
                                                                                                agent_num, batch_num)
        deserialized_step_info_list = self.__communication_manager.deserialize_info_list(step_results)

        return deserialized_char_info_list, deserialized_field_info_list, deserialized_step_info_list
